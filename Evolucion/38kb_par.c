enum { _clkfreq = 320000000 };
#include "38kb_weights.h"
#include "mnist_sample.h"


#define MODEL_NAME "38kb"
#define STACKSIZE 4096
#define WBUF_MAX_SIZE 72
#define _CLKFREQ 320000000

#define IN_H 28
#define IN_W 28
#define IN_C 1

#define C1_OUT 4
#define C1_K 3
#define C1_H 28
#define C1_W 28
#define C1_STRIDE 1

#define C2_OUT 8
#define C2_K 3
#define C2_H 14
#define C2_W 14
#define C2_STRIDE 2

#define C3_OUT 8
#define C3_K 3
#define C3_H 14
#define C3_W 14
#define C3_STRIDE 1

#define C4_OUT 12
#define C4_K 3
#define C4_H 7
#define C4_W 7
#define C4_STRIDE 2

#define GAP_OUT 12
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
    {7, 2},  // Conv1  (row axis)                                               
    {7, 2},  // Conv2  (row axis)                                               
    {7, 2},  // Conv3  (row axis)                                               
    {7, 2},  // Conv4  (row axis)                                               
    {3, 1},  // GAP  (channel axis)                                             
    {2, 1},  // Dense  (channel axis)                                           
};    
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

// ============================================================================
// PHASE HANDLERS - Split into separate functions to reduce code size
// ============================================================================

static void handle_conv1_chan(SharedNN *s, int start, int count, float *wbuf) {
    for (int j = 0; j < count; j++) {
        int oc = start + j;
        float local_bias = conv2d_biases[oc];
        for (int ky = 0; ky < C1_K; ky++)
            for (int kx = 0; kx < C1_K; kx++)
                wbuf[ky*C1_K+kx] = conv2d_weights[IDX_CONV1_W(ky,kx,0,oc)];
        for (int oy = 0; oy < C1_H; ++oy)
            for (int ox = 0; ox < C1_W; ++ox) {
                float sum = local_bias;
                for (int ky = 0; ky < C1_K; ky++) {
                    int iy = oy + ky - PAD1_H;
                    for (int kx = 0; kx < C1_K; kx++) {
                        int ix = ox + kx - PAD1_W;
                        if (iy >= 0 && iy < IN_H && ix >= 0 && ix < IN_W)
                            sum += s->img[iy][ix] * wbuf[ky*C1_K+kx];
                    }
                }
                s->c1_out[(oc*C1_H+oy)*C1_W+ox] = relu_f(sum);
            }
    }
}


static void handle_conv1_row(SharedNN *s, int start, int count, float *wbuf) {
    for (int oc = 0; oc < C1_OUT; oc++) {
        float local_bias = conv2d_biases[oc];
        for (int ky = 0; ky < C1_K; ky++)
            for (int kx = 0; kx < C1_K; kx++)
                wbuf[ky*C1_K+kx] = conv2d_weights[IDX_CONV1_W(ky,kx,0,oc)];
        for (int oy = start; oy < start+count; ++oy)
            for (int ox = 0; ox < C1_W; ++ox) {
                float sum = local_bias;
                for (int ky = 0; ky < C1_K; ky++) {
                    int iy = oy + ky - PAD1_H;
                    for (int kx = 0; kx < C1_K; kx++) {
                        int ix = ox + kx - PAD1_W;
                        if (iy >= 0 && iy < IN_H && ix >= 0 && ix < IN_W)
                            sum += s->img[iy][ix] * wbuf[ky*C1_K+kx];
                    }
                }
                s->c1_out[(oc*C1_H+oy)*C1_W+ox] = relu_f(sum);
            }
    }
}

static void handle_conv2_chan(SharedNN *s, int start, int count, float *wbuf) {
    for (int j = 0; j < count; j++) {
        int oc = start + j;
        float local_bias = conv2d_1_biases[oc];
        for (int ic = 0; ic < C1_OUT; ic++)
            for (int ky = 0; ky < C2_K; ky++)
                for (int kx = 0; kx < C2_K; kx++)
                    wbuf[ic*C2_K*C2_K + ky*C2_K + kx] =
                        conv2d_1_weights[IDX_CONV2_W(ky,kx,ic,oc)];
        for (int oy = 0; oy < C2_H; ++oy)
            for (int ox = 0; ox < C2_W; ++ox) {
                float sum = local_bias;
                for (int ic = 0; ic < C1_OUT; ic++) {
                    float *pw = &wbuf[ic*C2_K*C2_K];
                    for (int ky = 0; ky < C2_K; ky++) {
                        int iy = oy*C2_STRIDE + ky - PAD2_H;
                        for (int kx = 0; kx < C2_K; kx++) {
                            int ix = ox*C2_STRIDE + kx - PAD2_W;
                            if (iy >= 0 && iy < C1_H && ix >= 0 && ix < C1_W) {
                                int in_idx = (ic*C1_H+iy)*C1_W+ix;
                                sum += s->c1_out[in_idx] * pw[ky*C2_K+kx];
                            }
                        }
                    }
                }
                s->c2_out[(oc*C2_H+oy)*C2_W+ox] = relu_f(sum);
            }
    }
}

static void handle_conv2_row(SharedNN *s, int start, int count, float *wbuf) {
    for (int oc = 0; oc < C2_OUT; oc++) {
        float local_bias = conv2d_1_biases[oc];
        for (int ic = 0; ic < C1_OUT; ic++)
            for (int ky = 0; ky < C2_K; ky++)
                for (int kx = 0; kx < C2_K; kx++)
                    wbuf[ic*C2_K*C2_K + ky*C2_K + kx] =
                        conv2d_1_weights[IDX_CONV2_W(ky,kx,ic,oc)];
        for (int oy = start; oy < start+count; ++oy)
            for (int ox = 0; ox < C2_W; ++ox) {
                float sum = local_bias;
                for (int ic = 0; ic < C1_OUT; ic++) {
                    float *pw = &wbuf[ic*C2_K*C2_K];
                    for (int ky = 0; ky < C2_K; ky++) {
                        int iy = oy*C2_STRIDE + ky - PAD2_H;
                        for (int kx = 0; kx < C2_K; kx++) {
                            int ix = ox*C2_STRIDE + kx - PAD2_W;
                            if (iy >= 0 && iy < C1_H && ix >= 0 && ix < C1_W) {
                                int in_idx = (ic*C1_H+iy)*C1_W+ix;
                                sum += s->c1_out[in_idx] * pw[ky*C2_K+kx];
                            }
                        }
                    }
                }
                s->c2_out[(oc*C2_H+oy)*C2_W+ox] = relu_f(sum);
            }
    }
}

static void handle_conv3_chan(SharedNN *s, int start, int count, float *wbuf) {
    for (int j = 0; j < count; j++) {
        int oc = start + j;
        float local_bias = conv2d_2_biases[oc];
        for (int ic = 0; ic < C2_OUT; ic++)
            for (int ky = 0; ky < C3_K; ky++)
                for (int kx = 0; kx < C3_K; kx++)
                    wbuf[ic*C3_K*C3_K + ky*C3_K + kx] =
                        conv2d_2_weights[IDX_CONV3_W(ky,kx,ic,oc)];
        for (int oy = 0; oy < C3_H; ++oy)
            for (int ox = 0; ox < C3_W; ++ox) {
                float sum = local_bias;
                for (int ic = 0; ic < C2_OUT; ic++) {
                    float *pw = &wbuf[ic*C3_K*C3_K];
                    for (int ky = 0; ky < C3_K; ky++) {
                        int iy = oy + ky - PAD3_H;
                        for (int kx = 0; kx < C3_K; kx++) {
                            int ix = ox + kx - PAD3_W;
                            if (iy >= 0 && iy < C2_H && ix >= 0 && ix < C2_W) {
                                int in_idx = (ic*C2_H+iy)*C2_W+ix;
                                sum += s->c2_out[in_idx] * pw[ky*C3_K+kx];
                            }
                        }
                    }
                }
                s->c3_out[(oc*C3_H+oy)*C3_W+ox] = relu_f(sum);
            }
    }
}

static void handle_conv3_row(SharedNN *s, int start, int count, float *wbuf) {
    for (int oc = 0; oc < C3_OUT; oc++) {
        float local_bias = conv2d_2_biases[oc];
        for (int ic = 0; ic < C2_OUT; ic++)
            for (int ky = 0; ky < C3_K; ky++)
                for (int kx = 0; kx < C3_K; kx++)
                    wbuf[ic*C3_K*C3_K + ky*C3_K + kx] =
                        conv2d_2_weights[IDX_CONV3_W(ky,kx,ic,oc)];
        for (int oy = start; oy < start+count; ++oy)
            for (int ox = 0; ox < C3_W; ++ox) {
                float sum = local_bias;
                for (int ic = 0; ic < C2_OUT; ic++) {
                    float *pw = &wbuf[ic*C3_K*C3_K];
                    for (int ky = 0; ky < C3_K; ky++) {
                        int iy = oy + ky - PAD3_H;
                        for (int kx = 0; kx < C3_K; kx++) {
                            int ix = ox + kx - PAD3_W;
                            if (iy >= 0 && iy < C2_H && ix >= 0 && ix < C2_W) {
                                int in_idx = (ic*C2_H+iy)*C2_W+ix;
                                sum += s->c2_out[in_idx] * pw[ky*C3_K+kx];
                            }
                        }
                    }
                }
                s->c3_out[(oc*C3_H+oy)*C3_W+ox] = relu_f(sum);
            }
    }
}

static void handle_conv4_chan(SharedNN *s, int start, int count, float *wbuf) {
    for (int j = 0; j < count; j++) {
        int oc = start + j;
        float local_bias = conv2d_3_biases[oc];
        for (int ic = 0; ic < C3_OUT; ic++)
            for (int ky = 0; ky < C4_K; ky++)
                for (int kx = 0; kx < C4_K; kx++)
                    wbuf[ic*C4_K*C4_K + ky*C4_K + kx] =
                        conv2d_3_weights[IDX_CONV4_W(ky,kx,ic,oc)];
        for (int oy = 0; oy < C4_H; ++oy)
            for (int ox = 0; ox < C4_W; ++ox) {
                float sum = local_bias;
                for (int ic = 0; ic < C3_OUT; ic++) {
                    float *pw = &wbuf[ic*C4_K*C4_K];
                    for (int ky = 0; ky < C4_K; ky++) {
                        int iy = oy*C4_STRIDE + ky - PAD4_H;
                        for (int kx = 0; kx < C4_K; kx++) {
                            int ix = ox*C4_STRIDE + kx - PAD4_W;
                            if (iy >= 0 && iy < C3_H && ix >= 0 && ix < C3_W) {
                                int in_idx = (ic*C3_H+iy)*C3_W+ix;
                                sum += s->c3_out[in_idx] * pw[ky*C4_K+kx];
                            }
                        }
                    }
                }
                s->c4_out[(oc*C4_H+oy)*C4_W+ox] = relu_f(sum);
            }
    }
}

static void handle_conv4_row(SharedNN *s, int start, int count, float *wbuf) {
    for (int oc = 0; oc < C4_OUT; oc++) {
        float local_bias = conv2d_3_biases[oc];
        for (int ic = 0; ic < C3_OUT; ic++)
            for (int ky = 0; ky < C4_K; ky++)
                for (int kx = 0; kx < C4_K; kx++)
                    wbuf[ic*C4_K*C4_K + ky*C4_K + kx] =
                        conv2d_3_weights[IDX_CONV4_W(ky,kx,ic,oc)];
        for (int oy = start; oy < start+count; ++oy)
            for (int ox = 0; ox < C4_W; ++ox) {
                float sum = local_bias;
                for (int ic = 0; ic < C3_OUT; ic++) {
                    float *pw = &wbuf[ic*C4_K*C4_K];
                    for (int ky = 0; ky < C4_K; ky++) {
                        int iy = oy*C4_STRIDE + ky - PAD4_H;
                        for (int kx = 0; kx < C4_K; kx++) {
                            int ix = ox*C4_STRIDE + kx - PAD4_W;
                            if (iy >= 0 && iy < C3_H && ix >= 0 && ix < C3_W) {
                                int in_idx = (ic*C3_H+iy)*C3_W+ix;
                                sum += s->c3_out[in_idx] * pw[ky*C4_K+kx];
                            }
                        }
                    }
                }
                s->c4_out[(oc*C4_H+oy)*C4_W+ox] = relu_f(sum);
            }
    }
}

static void handle_gap(SharedNN *s, int start, int count) {
    for (int j = 0; j < count; j++) {
        int ch = start + j;
        float sum = 0.0f;
        for (int y = 0; y < C4_H; ++y)
            for (int x = 0; x < C4_W; ++x)
                sum += s->c4_out[(ch*C4_H+y)*C4_W+x];
        s->gap[ch] = sum / (C4_H * C4_W);
    }
}

static void handle_dense(SharedNN *s, int start, int count, float *wbuf) {
    for (int j = 0; j < count; j++) {
        int o = start + j;
        float local_bias = dense_biases[o];
        for (int i = 0; i < GAP_OUT; i++)
            wbuf[i] = dense_weights[IDX_DENSE_W(i,o)];
        float sum = local_bias;
        for (int i = 0; i < GAP_OUT; i++)
            sum += s->gap[i] * wbuf[i];
        s->out[o] = sum;
    }
}

// ============================================================================
// MAIN WORKER - Small dispatcher that calls phase handlers
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
                    if (axis == 0) handle_conv1_chan(s, start, count, wbuf);
                    else handle_conv1_row(s, start, count, wbuf);
                    break;
                case 2:
                    if (axis == 0) handle_conv2_chan(s, start, count, wbuf);
                    else handle_conv2_row(s, start, count, wbuf);
                    break;
                case 3:
                    if (axis == 0) handle_conv3_chan(s, start, count, wbuf);
                    else handle_conv3_row(s, start, count, wbuf);
                    break;
                case 4:
                    if (axis == 0) handle_conv4_chan(s, start, count, wbuf);
                    else handle_conv4_row(s, start, count, wbuf);
                    break;
                case 5:
                    handle_gap(s, start, count);
                    break;
                case 6:
                    handle_dense(s, start, count, wbuf);
                    break;
            }
        }

        s->worker_done[id] = expected_phase;

        while (s->phase == expected_phase) {
            _waitx(_clockfreq() / 2000);
        }
    }
}

// ============================================================================
// SUPERVISOR
// ============================================================================

int main() {
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
        const LayerDesc *ld = &LAYER_PROFILE[ph-1];
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
    uint32_t us_total = t1 - t0;

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
    printf("Clock freq: %lu Hz\n", _clockfreq());

    NN.phase = -1;
    _waitx(_clockfreq() / 100);
    for (int i = 0; i < MAX_WORKERS; ++i) _cogstop(cid_workers[i]);

    for (;;) _waitx(_clockfreq() / 2000);
}
