enum { _clkfreq = 320000000 };
#include "240kb_weights.h"
#include "mnist_sample.h"

#define MODEL_NAME    "240kb"
#define STACKSIZE     4096
#define WBUF_MAX_SIZE 324

#define IN_H 28
#define IN_W 28
#define IN_C 1

#define C1_OUT 18
#define C1_K   3
#define C1_H   28
#define C1_W   28
#define C1_STRIDE 1
#define PAD1_H 1
#define PAD1_W 1

#define C2_OUT 36
#define C2_K   3
#define C2_H   14
#define C2_W   14
#define C2_STRIDE 2
#define PAD2_H 0
#define PAD2_W 0

#define C3_OUT 36
#define C3_K   3
#define C3_H   14
#define C3_W   14
#define C3_STRIDE 1
#define PAD3_H 1
#define PAD3_W 1

#define C4_OUT 36
#define C4_K   3
#define C4_H   7
#define C4_W   7
#define C4_STRIDE 2
#define PAD4_H 0
#define PAD4_W 0

#define GAP_OUT   36
#define DENSE_OUT 10
#define MAX_WORKERS 7

// ============================================================================
// LAYER PROFILE  -- edit num_workers (1-7) and strategy (1=channel, 2=row)
// per layer to tune parallelism.  Run bench_240kb.c on hardware first to
// find the optimal values; paste the printed LAYER_PROFILE here.
// ============================================================================
typedef struct {
    int num_workers;
    int strategy;   // 1 = partition output channels across workers
                    // 2 = partition output rows across workers
} LayerDesc;

static const LayerDesc LAYER_PROFILE[6] = {
    {7, 1},  // Conv1  (18 output channels, 28x28)
    {7, 1},  // Conv2  (36 output channels, 14x14, stride 2)
    {7, 1},  // Conv3  (36 output channels, 14x14)
    {7, 1},  // Conv4  (36 output channels, 7x7, stride 2)
    {7, 1},  // GAP    (36 channels)
    {7, 1},  // Dense  (36 -> 10)
};
// ====================== END MODEL CONFIGURATION BLOCK =======================

#include <stdio.h>
#include <propeller2.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

// ============================================================================
// SHARED MEMORY
// ============================================================================
typedef struct {
    volatile float img[IN_H][IN_W];
    volatile float c1_out[C1_OUT * C1_H * C1_W];
    volatile float c2_out[C2_OUT * C2_H * C2_W];
    volatile float c3_out[C3_OUT * C3_H * C3_W];
    volatile float c4_out[C4_OUT * C4_H * C4_W];
    volatile float gap[GAP_OUT];
    volatile float out[DENSE_OUT];

    volatile int phase;           // monotonically increasing dispatch counter
    volatile int active_workers;  // how many cogs are doing work this phase
    volatile int worker_done[MAX_WORKERS];
    volatile int layer_start[MAX_WORKERS];  // first unit (channel or row) for worker i
    volatile int layer_count[MAX_WORKERS];  // number of units for worker i
    volatile int partition_axis;            // 0=channel, 1=row
} SharedNN;

SharedNN NN;

typedef struct {
    SharedNN *shared;
    int id;
} WorkerParam;

static unsigned char stacks_workers[MAX_WORKERS][STACKSIZE];

// ============================================================================
// MATH HELPERS
// ============================================================================
static inline float relu_f(float x) { return x < 0.0f ? 0.0f : x; }

static void softmax_f(float *x, int len) {
    float maxv = x[0];
    for (int i = 1; i < len; ++i) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) { x[i] = expf(x[i] - maxv); sum += x[i]; }
    for (int i = 0; i < len; ++i) x[i] /= sum;
}

static inline void get_worker_range(int total, int id, int nw,
                                    int *start, int *count) {
    int base = total / nw, rem = total % nw;
    *start = id * base + (id < rem ? id : rem);
    *count = base + (id < rem ? 1 : 0);
}

// ============================================================================
// GENERIC LAYER FUNCTIONS
// ============================================================================

// 2-D convolution -- supports both channel-axis and row-axis partitioning.
static void conv2d_layer(volatile float *input, volatile float *output,
                         int in_c, int in_h, int in_w,
                         int out_c, int out_h, int out_w,
                         int k, int stride, int pad_h, int pad_w,
                         const float *weights, const float *bias,
                         int start, int count, int axis, float *wbuf)
{
    if (axis == 0) {                    /* partition by output channel */
        for (int j = 0; j < count; j++) {
            int oc = start + j;
            float lb = bias[oc];
            for (int ic = 0; ic < in_c; ic++)
                for (int ky = 0; ky < k; ky++)
                    for (int kx = 0; kx < k; kx++)
                        wbuf[ic*k*k + ky*k + kx] =
                            weights[((ky*k + kx)*in_c + ic)*out_c + oc];
            for (int oy = 0; oy < out_h; ++oy)
                for (int ox = 0; ox < out_w; ++ox) {
                    float sum = lb;
                    for (int ic = 0; ic < in_c; ic++) {
                        float *pw = &wbuf[ic*k*k];
                        for (int ky = 0; ky < k; ky++) {
                            int iy = oy*stride + ky - pad_h;
                            for (int kx = 0; kx < k; kx++) {
                                int ix = ox*stride + kx - pad_w;
                                if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w)
                                    sum += input[(ic*in_h+iy)*in_w+ix]
                                           * pw[ky*k+kx];
                            }
                        }
                    }
                    output[(oc*out_h+oy)*out_w+ox] = relu_f(sum);
                }
        }
    } else {                           /* partition by output row */
        for (int oc = 0; oc < out_c; oc++) {
            float lb = bias[oc];
            for (int ic = 0; ic < in_c; ic++)
                for (int ky = 0; ky < k; ky++)
                    for (int kx = 0; kx < k; kx++)
                        wbuf[ic*k*k + ky*k + kx] =
                            weights[((ky*k + kx)*in_c + ic)*out_c + oc];
            for (int oy = start; oy < start + count; ++oy)
                for (int ox = 0; ox < out_w; ++ox) {
                    float sum = lb;
                    for (int ic = 0; ic < in_c; ic++) {
                        float *pw = &wbuf[ic*k*k];
                        for (int ky = 0; ky < k; ky++) {
                            int iy = oy*stride + ky - pad_h;
                            for (int kx = 0; kx < k; kx++) {
                                int ix = ox*stride + kx - pad_w;
                                if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w)
                                    sum += input[(ic*in_h+iy)*in_w+ix]
                                           * pw[ky*k+kx];
                            }
                        }
                    }
                    output[(oc*out_h+oy)*out_w+ox] = relu_f(sum);
                }
        }
    }
}

static void gap_layer(volatile float *input, volatile float *output,
                      int c, int h, int w, int start, int count)
{
    float sp = (float)(h * w);
    for (int j = 0; j < count; j++) {
        int ch = start + j;
        float sum = 0.0f;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                sum += input[(ch*h+y)*w+x];
        output[ch] = sum / sp;
    }
}

static void dense_layer(volatile float *input, volatile float *output,
                        int in_size, int out_size,
                        const float *weights, const float *bias,
                        int start, int count, float *wbuf)
{
    for (int j = 0; j < count; j++) {
        int o = start + j;
        float lb = bias[o];
        for (int i = 0; i < in_size; i++)
            wbuf[i] = weights[i*out_size + o];
        float sum = lb;
        for (int i = 0; i < in_size; i++)
            sum += input[i] * wbuf[i];
        output[o] = sum;
    }
}

// ============================================================================
// WORKER COG
// Spins waiting for the phase counter to advance, then dispatches to the
// appropriate layer function using its assigned start/count range.
// Idle workers (id >= active_workers) skip computation and signal done.
// ============================================================================
void cog_worker(void *p) {
    WorkerParam *wp = (WorkerParam *)p;
    SharedNN *s = wp->shared;
    int id = wp->id;

    #if WBUF_MAX_SIZE <= 512
    __attribute__((cog)) float wbuf[WBUF_MAX_SIZE];
    #else
    float wbuf[WBUF_MAX_SIZE];
    #endif

    int last_phase = 0;

    while (1) {
        // Wait for the supervisor to advance the phase counter
        int cur;
        do {
            cur = s->phase;
            if (cur == -1) return;   // shutdown signal
            if (cur != last_phase) break;
            _waitx(_clockfreq() / 2000);
        } while (1);
        last_phase = cur;

        if (id < s->active_workers) {
            int start = s->layer_start[id];
            int count = s->layer_count[id];
            int axis  = s->partition_axis;

            if (count > 0) {
                switch (cur) {
                    case 1:
                        conv2d_layer((volatile float*)s->img, s->c1_out,
                                     IN_C, IN_H, IN_W, C1_OUT, C1_H, C1_W,
                                     C1_K, C1_STRIDE, PAD1_H, PAD1_W,
                                     conv2d_weights, conv2d_biases,
                                     start, count, axis, wbuf);
                        break;
                    case 2:
                        conv2d_layer(s->c1_out, s->c2_out,
                                     C1_OUT, C1_H, C1_W, C2_OUT, C2_H, C2_W,
                                     C2_K, C2_STRIDE, PAD2_H, PAD2_W,
                                     conv2d_1_weights, conv2d_1_biases,
                                     start, count, axis, wbuf);
                        break;
                    case 3:
                        conv2d_layer(s->c2_out, s->c3_out,
                                     C2_OUT, C2_H, C2_W, C3_OUT, C3_H, C3_W,
                                     C3_K, C3_STRIDE, PAD3_H, PAD3_W,
                                     conv2d_2_weights, conv2d_2_biases,
                                     start, count, axis, wbuf);
                        break;
                    case 4:
                        conv2d_layer(s->c3_out, s->c4_out,
                                     C3_OUT, C3_H, C3_W, C4_OUT, C4_H, C4_W,
                                     C4_K, C4_STRIDE, PAD4_H, PAD4_W,
                                     conv2d_3_weights, conv2d_3_biases,
                                     start, count, axis, wbuf);
                        break;
                    case 5:
                        gap_layer(s->c4_out, s->gap,
                                  C4_OUT, C4_H, C4_W, start, count);
                        break;
                    case 6:
                        dense_layer(s->gap, s->out,
                                    GAP_OUT, DENSE_OUT,
                                    dense_weights, dense_biases,
                                    start, count, wbuf);
                        break;
                }
            }
        }
        // Signal completion for this phase (active and idle workers both signal)
        s->worker_done[id] = cur;
    }
}

// ============================================================================
// SUPERVISOR
// ============================================================================
int main(void) {
    printf("\n=== Dynamic Parallel MNIST CNN --- %s ===\n", MODEL_NAME);
    printf("Model size: %d parameters\n", TOTAL_PARAMS);
    printf("Sample index: %d, True label: %d\n\n", SAMPLE_INDEX, SAMPLE_LABEL);

    memset((void *)&NN, 0, sizeof(NN));
    NN.phase = 0;

    for (int y = 0; y < IN_H; ++y)
        for (int x = 0; x < IN_W; ++x)
            NN.img[y][x] = mnist_sample[y][x];

    static WorkerParam wparams[MAX_WORKERS];
    int cid[MAX_WORKERS];
    for (int i = 0; i < MAX_WORKERS; i++) {
        wparams[i].shared = &NN;
        wparams[i].id     = i;
        cid[i] = _cogstart_C(cog_worker, &wparams[i],
                             stacks_workers[i], STACKSIZE);
    }

    uint32_t t0 = _getus();

    for (int ph = 1; ph <= 6; ph++) {
        const LayerDesc *ld = &LAYER_PROFILE[ph-1];
        int nw    = ld->num_workers;
        int strat = ld->strategy;

        // Determine the axis size for this layer
        int out_c = 0, out_h = 0;
        switch (ph) {
            case 1: out_c = C1_OUT;    out_h = C1_H; break;
            case 2: out_c = C2_OUT;    out_h = C2_H; break;
            case 3: out_c = C3_OUT;    out_h = C3_H; break;
            case 4: out_c = C4_OUT;    out_h = C4_H; break;
            case 5: out_c = GAP_OUT;   out_h = 1;    break;
            case 6: out_c = DENSE_OUT; out_h = 1;    break;
        }

        int total_units   = (strat == 1) ? out_c : out_h;
        NN.partition_axis = (strat == 1) ? 0 : 1;
        NN.active_workers = nw;

        for (int i = 0; i < MAX_WORKERS; i++) {
            if (i < nw)
                get_worker_range(total_units, i, nw,
                                 &NN.layer_start[i], &NN.layer_count[i]);
            else {
                NN.layer_start[i] = 0;
                NN.layer_count[i] = 0;
            }
        }

        for (int i = 0; i < MAX_WORKERS; i++) NN.worker_done[i] = 0;
        NN.phase = ph;

        // Wait for all workers (active and idle) to acknowledge
        int done;
        do {
            done = 1;
            for (int i = 0; i < MAX_WORKERS; i++)
                if (NN.worker_done[i] != ph) { done = 0; break; }
            if (!done) _waitx(_clockfreq() / 2000);
        } while (!done);
    }

    uint32_t t1 = _getus();
    uint32_t us_total = t1 - t0;

    float logits[DENSE_OUT];
    for (int k = 0; k < DENSE_OUT; ++k) logits[k] = NN.out[k];
    softmax_f(logits, DENSE_OUT);

    int pred = 0;
    for (int i = 1; i < DENSE_OUT; ++i)
        if (logits[i] > logits[pred]) pred = i;

    printf("\nProbabilities:\n");
    for (int i = 0; i < DENSE_OUT; ++i) {
        printf("%d: %.6f", i, logits[i]);
        if (i == pred)         printf("  <-- PREDICTION");
        if (i == SAMPLE_LABEL) printf("  (TRUE LABEL)");
        printf("\n");
    }
    printf("\nPredicted digit: %d\n", pred);
    printf("True digit:      %d\n", SAMPLE_LABEL);
    printf("Correct:         %s\n", (pred == SAMPLE_LABEL) ? "YES" : "NO");
    printf("Inference time:  %lu.%03lu ms\n",
           us_total / 1000, us_total % 1000);
    printf("Clock freq:      %lu Hz\n", (uint32_t)_clockfreq());

    NN.phase = -1;
    _waitx(_clockfreq() / 100);
    for (int i = 0; i < MAX_WORKERS; i++) _cogstop(cid[i]);

    for (;;) _waitx(_clockfreq() / 2000);
}
