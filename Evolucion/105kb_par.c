enum { _clkfreq = 320000000 };
#include "105kb_weights.h"
#include "mnist_sample.h"

#define MODEL_NAME "105kb"
#define STACKSIZE 4096
#define WBUF_MAX_SIZE 180

#define IN_H 28
#define IN_W 28

#define IN_C 1

#define C1_OUT 10
#define C1_K 3
#define C1_H 28
#define C1_W 28
#define C1_STRIDE 1

#define C2_OUT 20
#define C2_K 3
#define C2_H 14
#define C2_W 14
#define C2_STRIDE 2

#define C3_OUT 20
#define C3_K 3
#define C3_H 14
#define C3_W 14
#define C3_STRIDE 1

#define C4_OUT 20
#define C4_K 3
#define C4_H 7
#define C4_W 7
#define C4_STRIDE 2

#define GAP_OUT 20
#define DENSE_OUT 10

#define PAD1_H 1
#define PAD1_W 1
#define PAD2_H 0
#define PAD2_W 0
#define PAD3_H 1
#define PAD3_W 1
#define PAD4_H 0
#define PAD4_W 0

#define IDX_CONV1_W(ky,kx,ic,oc) ((((ky)*C1_K+(kx))*IN_C+(ic))*C1_OUT+(oc))
#define IDX_CONV2_W(ky,kx,ic,oc) ((((ky)*C2_K+(kx))*C1_OUT+(ic))*C2_OUT+(oc))
#define IDX_CONV3_W(ky,kx,ic,oc) ((((ky)*C3_K+(kx))*C2_OUT+(ic))*C3_OUT+(oc))
#define IDX_CONV4_W(ky,kx,ic,oc) ((((ky)*C4_K+(kx))*C3_OUT+(ic))*C4_OUT+(oc))
#define IDX_DENSE_W(i,o) ((i)*DENSE_OUT+(o))

typedef struct {
    int num_workers;
    int strategy;
} LayerDesc;

static const LayerDesc LAYER_PROFILE[6] = {                                     
    {7, 2},  // Conv1 (row axis)                                                
    {7, 2},  // Conv2 (channel axis)                                            
    {7, 1},  // Conv3 (channel axis)                                            
    {7, 1},  // Conv4 (channel axis)                                            
    {5, 1},  // GAP (channel axis)                                              
    {7, 1},  // Dense (channel axis)                                            
};  
// ====================== END MODEL CONFIGURATION BLOCK =======================
// ====================== END MODEL CONFIGURATION BLOCK =======================

#include <stdio.h>
#include <propeller2.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define MAX_WORKERS 7

typedef struct {
    volatile float img[IN_H][IN_W];
    volatile float c1_out[C1_OUT * C1_H * C1_W];
    volatile float c2_out[C2_OUT * C2_H * C2_W];
    volatile float c3_out[C3_OUT * C3_H * C3_W];
    volatile float c4_out[C4_OUT * C4_H * C4_W];
    volatile float gap[GAP_OUT];
    volatile float out[DENSE_OUT];

    volatile int phase;
    volatile int active_workers;
    volatile int worker_done[MAX_WORKERS];
    volatile int layer_start[MAX_WORKERS];
    volatile int layer_count[MAX_WORKERS];
    volatile int partition_axis;
} SharedNN;

SharedNN NN;

typedef struct {
    SharedNN *shared;
    int id;
} WorkerParam;

static unsigned char stacks_workers[MAX_WORKERS][STACKSIZE];

static inline float relu_f(float x) { return x < 0.0f ? 0.0f : x; }

static inline void get_worker_range(int total, int id, int nw, int *start, int *count) {
    int base = total / nw;
    int rem = total % nw;
    *start = id * base + (id < rem ? id : rem);
    *count = base + (id < rem ? 1 : 0);
}

/* ============================================================================
   GENERIC LAYER FUNCTIONS - One implementation, called per-layer
   ============================================================================ */

static void conv2d_layer(volatile float *input, volatile float *output,
                         int in_c, int in_h, int in_w,
                         int out_c, int out_h, int out_w,
                         int k, int stride, int pad_h, int pad_w,
                         const float *weights, const float *bias,
                         int start, int count, int axis, float *wbuf)
{
    if (axis == 0) {                     /* channel-wise */
        for (int j = 0; j < count; j++) {
            int oc = start + j;
            float local_bias = bias[oc];
            for (int ic = 0; ic < in_c; ic++)
                for (int ky = 0; ky < k; ky++)
                    for (int kx = 0; kx < k; kx++)
                        wbuf[ic*k*k + ky*k + kx] =
                            weights[((ky*k + kx)*in_c + ic)*out_c + oc];

            for (int oy = 0; oy < out_h; ++oy)
                for (int ox = 0; ox < out_w; ++ox) {
                    float sum = local_bias;
                    for (int ic = 0; ic < in_c; ic++) {
                        float *pw = &wbuf[ic*k*k];
                        for (int ky = 0; ky < k; ky++) {
                            int iy = oy*stride + ky - pad_h;
                            for (int kx = 0; kx < k; kx++) {
                                int ix = ox*stride + kx - pad_w;
                                if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w) {
                                    int in_idx = (ic*in_h + iy)*in_w + ix;
                                    sum += input[in_idx] * pw[ky*k + kx];
                                }
                            }
                        }
                    }
                    output[(oc*out_h + oy)*out_w + ox] = relu_f(sum);
                }
        }
    } else {                             /* row-wise fallback */
        for (int oc = 0; oc < out_c; oc++) {
            float local_bias = bias[oc];
            for (int ic = 0; ic < in_c; ic++)
                for (int ky = 0; ky < k; ky++)
                    for (int kx = 0; kx < k; kx++)
                        wbuf[ic*k*k + ky*k + kx] =
                            weights[((ky*k + kx)*in_c + ic)*out_c + oc];

            for (int oy = start; oy < start + count; ++oy)
                for (int ox = 0; ox < out_w; ++ox) {
                    float sum = local_bias;
                    for (int ic = 0; ic < in_c; ic++) {
                        float *pw = &wbuf[ic*k*k];
                        for (int ky = 0; ky < k; ky++) {
                            int iy = oy*stride + ky - pad_h;
                            for (int kx = 0; kx < k; kx++) {
                                int ix = ox*stride + kx - pad_w;
                                if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w) {
                                    int in_idx = (ic*in_h + iy)*in_w + ix;
                                    sum += input[in_idx] * pw[ky*k + kx];
                                }
                            }
                        }
                    }
                    output[(oc*out_h + oy)*out_w + ox] = relu_f(sum);
                }
        }
    }
}

static void gap_layer(volatile float *input, volatile float *output,
                      int c, int h, int w, int start, int count)
{
    float spatial = (float)(h * w);
    for (int j = 0; j < count; j++) {
        int ch = start + j;
        float sum = 0.0f;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                sum += input[(ch*h + y)*w + x];
        output[ch] = sum / spatial;
    }
}

static void dense_layer_worker(volatile float *input, volatile float *output,
                               int in_size, int out_size,
                               const float *weights, const float *bias,
                               int start, int count, float *wbuf)
{
    for (int j = 0; j < count; j++) {
        int o = start + j;
        float local_bias = bias[o];
        for (int i = 0; i < in_size; i++)
            wbuf[i] = weights[i*out_size + o];
        float sum = local_bias;
        for (int i = 0; i < in_size; i++)
            sum += input[i] * wbuf[i];
        output[o] = sum;
    }
}

/* ============================================================================
   WORKER COG  - tiny dispatch loop, generic functions live in HUB
   ============================================================================ */

void cog_worker(void *p)
{
    WorkerParam *wp = (WorkerParam *)p;
    SharedNN *s = wp->shared;
    int id = wp->id;
    float wbuf[WBUF_MAX_SIZE];

    while (1) {
        int expected_phase = s->phase;
        if (expected_phase == -1) return;

        if (id >= s->active_workers) {
            _waitx(_clockfreq() / 2000);
            continue;
        }

        int start = s->layer_start[id];
        int count = s->layer_count[id];
        int axis  = s->partition_axis;

        if (count > 0) {
            switch (expected_phase) {
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
                    dense_layer_worker(s->gap, s->out,
                                       GAP_OUT, DENSE_OUT,
                                       dense_weights, dense_biases,
                                       start, count, wbuf);
                    break;
            }
        }

        s->worker_done[id] = expected_phase;

        while (s->phase == expected_phase)
            _waitx(_clockfreq() / 2000);
    }
}

/* ============================================================================
   SUPERVISOR
   ============================================================================ */

int main()
{
    printf("\n=== Dynamic Parallel MNIST CNN --- %s ===\n", MODEL_NAME);
    printf("Model size: %d parameters\n", TOTAL_PARAMS);
    printf("Sample index: %d, True label: %d\n\n", SAMPLE_INDEX, SAMPLE_LABEL);

    memset((void *)&NN, 0, sizeof(NN));
    NN.phase = 0;

    for (int y = 0; y < IN_H; ++y)
        for (int x = 0; x < IN_W; ++x)
            NN.img[y][x] = mnist_sample[y][x];

    static WorkerParam wparams[MAX_WORKERS];
    int cid_workers[MAX_WORKERS];
    for (int i = 0; i < MAX_WORKERS; ++i) {
        wparams[i].shared = &NN;
        wparams[i].id = i;
        cid_workers[i] = _cogstart_C(cog_worker, &wparams[i],
                                     stacks_workers[i], STACKSIZE);
    }

    uint32_t t0 = _getus();

    for (int ph = 1; ph <= 6; ph++) {
        const LayerDesc *ld = &LAYER_PROFILE[ph - 1];
        int nw = ld->num_workers;
        int strategy = ld->strategy;

        int out_c = 0, out_h = 0;
        switch (ph) {
            case 1: out_c = C1_OUT; out_h = C1_H; break;
            case 2: out_c = C2_OUT; out_h = C2_H; break;
            case 3: out_c = C3_OUT; out_h = C3_H; break;
            case 4: out_c = C4_OUT; out_h = C4_H; break;
            case 5: out_c = GAP_OUT; out_h = 1; break;
            case 6: out_c = DENSE_OUT; out_h = 1; break;
        }

        if (strategy == 0) strategy = (out_c >= nw) ? 1 : 2;
        int total_units = (strategy == 1) ? out_c : out_h;
        NN.partition_axis = (strategy == 1) ? 0 : 1;

        for (int i = 0; i < MAX_WORKERS; i++) {
            if (i < nw) {
                int lstart, lcnt;
                get_worker_range(total_units, i, nw, &lstart, &lcnt);
                NN.layer_start[i] = lstart;
                NN.layer_count[i] = lcnt;
            } else {
                NN.layer_start[i] = 0;
                NN.layer_count[i] = 0;
            }
        }

        for (int i = 0; i < MAX_WORKERS; ++i) NN.worker_done[i] = 0;
        NN.active_workers = nw;
        NN.phase = ph;

        int done;
        do {
            done = 1;
            for (int i = 0; i < nw; i++) {
                if (NN.worker_done[i] != ph) { done = 0; break; }
            }
            if (!done) _waitx(_clockfreq() / 2000);
        } while (!done);
    }

    uint32_t t1 = _getus();
    uint32_t us_total = t1-t0;

    // Find predicted class directly from raw outputs (no softmax)
int pred = 0;
for (int i = 1; i < DENSE_OUT; ++i)
    if (NN.out[i] > NN.out[pred]) pred = i;

printf("\nRaw outputs (logits):\n");
for (int i = 0; i < DENSE_OUT; ++i) {
    printf("%d: %f", i, (double)NN.out[i]);
    if (i == pred) printf("  <-- PREDICTION");
    if (i == SAMPLE_LABEL) printf("  (TRUE LABEL)");
    printf("\n");
}
    printf("\nPredicted digit: %d\n", pred);
    printf("True digit: %d\n", SAMPLE_LABEL);
    printf("Correct: %s\n", (pred == SAMPLE_LABEL) ? "YES" : "NO");
    printf("Inference time: %lu.%01lu ms\n", us_total / 1000, us_total%1000);
    
    NN.phase = -1;
    _waitx(_clockfreq() / 100);
    for (int i = 0; i < MAX_WORKERS; ++i) _cogstop(cid_workers[i]);

    for (;;) _waitx(_clockfreq() / 2000);
}
