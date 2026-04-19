// ========================= MODEL CONFIGURATION BLOCK ========================
#include "320kb_weights.h"
#include "mnist_sample.h"

#define MODEL_NAME "320kb"
#define STACKSIZE 4096

#define IN_H 28
#define IN_W 28
#define IN_C 1

#define C1_OUT 22
#define C1_K 3
#define C1_H 28
#define C1_W 28
#define C1_STRIDE 1

#define C2_OUT 44
#define C2_K 3
#define C2_H 14
#define C2_W 14
#define C2_STRIDE 2

#define C3_OUT 44
#define C3_K 3
#define C3_H 14
#define C3_W 14
#define C3_STRIDE 1

#define C4_OUT 44
#define C4_K 3
#define C4_H 7
#define C4_W 7
#define C4_STRIDE 2

#define GAP_OUT 44
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

#define N_WORKERS 7
// ====================== END MODEL CONFIGURATION BLOCK =======================

#include <stdio.h>
#include <propeller2.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

unsigned char stacks_workers[N_WORKERS][STACKSIZE];

typedef struct {
    volatile float img[IN_H][IN_W];
    volatile float c1_out[C1_OUT * C1_H * C1_W];
    volatile float c2_out[C2_OUT * C2_H * C2_W];
    volatile float c3_out[C3_OUT * C3_H * C3_W];
    volatile float c4_out[C4_OUT * C4_H * C4_W];
    volatile float gap[GAP_OUT];
    volatile float out[DENSE_OUT];
    volatile int phase;
    volatile int worker_done[N_WORKERS];
} SharedNN;

SharedNN NN;

typedef struct {
    SharedNN *shared;
    int id;
} WorkerParam;

static inline float relu_f(float x) { return x < 0.0f ? 0.0f : x; }

static void softmax_f(float *x, int len) {
    float maxv = x[0];
    for (int i = 1; i < len; ++i) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) { x[i] = expf(x[i] - maxv); sum += x[i]; }
    for (int i = 0; i < len; ++i) x[i] /= sum;
}

static inline void get_worker_range(int total, int id, int nw, int *start, int *count) {
    int base = total / nw;
    int rem = total % nw;
    *start = id * base + (id < rem ? id : rem);
    *count = base + (id < rem ? 1 : 0);
}

void cog_worker(void *p) {
    WorkerParam *wp = (WorkerParam *)p;
    SharedNN *s = wp->shared;
    int id = wp->id;

    int c1_start, c1_cnt;
    int c2_start, c2_cnt;
    int c3_start, c3_cnt;
    int c4_start, c4_cnt;
    int gap_start, gap_cnt;
    int dense_start, dense_cnt;

    get_worker_range(C1_OUT, id, N_WORKERS, &c1_start, &c1_cnt);
    get_worker_range(C2_OUT, id, N_WORKERS, &c2_start, &c2_cnt);
    get_worker_range(C3_OUT, id, N_WORKERS, &c3_start, &c3_cnt);
    get_worker_range(C4_OUT, id, N_WORKERS, &c4_start, &c4_cnt);
    get_worker_range(GAP_OUT, id, N_WORKERS, &gap_start, &gap_cnt);
    get_worker_range(DENSE_OUT, id, N_WORKERS, &dense_start, &dense_cnt);

    __attribute__((cog)) float channel_weights[396];
    __attribute__((cog)) float local_bias;

    for (int expect = 1; expect <= 6; expect++) {
        while (s->phase != expect) _waitx(_clockfreq() / 2000);

        if (expect == 1) {
            for (int j = 0; j < c1_cnt; j++) {
                int oc = c1_start + j;
                local_bias = conv2d_biases[oc];
                for (int ky = 0; ky < C1_K; ky++) {
                    for (int kx = 0; kx < C1_K; kx++) {
                        int idx = IDX_CONV1_W(ky, kx, 0, oc);
                        channel_weights[ky*C1_K + kx] = conv2d_weights[idx];
                    }
                }
                for (int oy = 0; oy < C1_H; ++oy) {
                    for (int ox = 0; ox < C1_W; ++ox) {
                        float sum = local_bias;
                        for (int ky = 0; ky < C1_K; ky++) {
                            int iy = oy + ky - PAD1_H;
                            for (int kx = 0; kx < C1_K; kx++) {
                                int ix = ox + kx - PAD1_W;
                                if (iy >= 0 && iy < IN_H && ix >= 0 && ix < IN_W) {
                                    sum += s->img[iy][ix] * channel_weights[ky*C1_K + kx];
                                }
                            }
                        }
                        int out_idx = (oc * C1_H + oy) * C1_W + ox;
                        s->c1_out[out_idx] = relu_f(sum);
                    }
                }
            }
            s->worker_done[id] = 1;
        }

        if (expect == 2) {
            for (int j = 0; j < c2_cnt; j++) {
                int oc = c2_start + j;
                local_bias = conv2d_1_biases[oc];
                for (int ic = 0; ic < C1_OUT; ic++) {
                    for (int ky = 0; ky < C2_K; ky++) {
                        for (int kx = 0; kx < C2_K; kx++) {
                            int idx = IDX_CONV2_W(ky, kx, ic, oc);
                            channel_weights[ic * C2_K*C2_K + ky*C2_K + kx] = conv2d_1_weights[idx];
                        }
                    }
                }
                for (int oy = 0; oy < C2_H; ++oy) {
                    for (int ox = 0; ox < C2_W; ++ox) {
                        float sum = local_bias;
                        for (int ic = 0; ic < C1_OUT; ic++) {
                            float *w = &channel_weights[ic * C2_K*C2_K];
                            for (int ky = 0; ky < C2_K; ky++) {
                                int iy = oy * C2_STRIDE + ky - PAD2_H;
                                for (int kx = 0; kx < C2_K; kx++) {
                                    int ix = ox * C2_STRIDE + kx - PAD2_W;
                                    if (iy >= 0 && iy < C1_H && ix >= 0 && ix < C1_W) {
                                        int in_idx = (ic * C1_H + iy) * C1_W + ix;
                                        sum += s->c1_out[in_idx] * w[ky*C2_K + kx];
                                    }
                                }
                            }
                        }
                        int out_idx = (oc * C2_H + oy) * C2_W + ox;
                        s->c2_out[out_idx] = relu_f(sum);
                    }
                }
            }
            s->worker_done[id] = 2;
        }

        if (expect == 3) {
            for (int j = 0; j < c3_cnt; j++) {
                int oc = c3_start + j;
                local_bias = conv2d_2_biases[oc];
                for (int ic = 0; ic < C2_OUT; ic++) {
                    for (int ky = 0; ky < C3_K; ky++) {
                        for (int kx = 0; kx < C3_K; kx++) {
                            int idx = IDX_CONV3_W(ky, kx, ic, oc);
                            channel_weights[ic * C3_K*C3_K + ky*C3_K + kx] = conv2d_2_weights[idx];
                        }
                    }
                }
                for (int oy = 0; oy < C3_H; ++oy) {
                    for (int ox = 0; ox < C3_W; ++ox) {
                        float sum = local_bias;
                        for (int ic = 0; ic < C2_OUT; ic++) {
                            float *w = &channel_weights[ic * C3_K*C3_K];
                            for (int ky = 0; ky < C3_K; ky++) {
                                int iy = oy + ky - PAD3_H;
                                for (int kx = 0; kx < C3_K; kx++) {
                                    int ix = ox + kx - PAD3_W;
                                    if (iy >= 0 && iy < C2_H && ix >= 0 && ix < C2_W) {
                                        int in_idx = (ic * C2_H + iy) * C2_W + ix;
                                        sum += s->c2_out[in_idx] * w[ky*C3_K + kx];
                                    }
                                }
                            }
                        }
                        int out_idx = (oc * C3_H + oy) * C3_W + ox;
                        s->c3_out[out_idx] = relu_f(sum);
                    }
                }
            }
            s->worker_done[id] = 3;
        }

        if (expect == 4) {
            for (int j = 0; j < c4_cnt; j++) {
                int oc = c4_start + j;
                local_bias = conv2d_3_biases[oc];
                for (int ic = 0; ic < C3_OUT; ic++) {
                    for (int ky = 0; ky < C4_K; ky++) {
                        for (int kx = 0; kx < C4_K; kx++) {
                            int idx = IDX_CONV4_W(ky, kx, ic, oc);
                            channel_weights[ic * C4_K*C4_K + ky*C4_K + kx] = conv2d_3_weights[idx];
                        }
                    }
                }
                for (int oy = 0; oy < C4_H; ++oy) {
                    for (int ox = 0; ox < C4_W; ++ox) {
                        float sum = local_bias;
                        for (int ic = 0; ic < C3_OUT; ic++) {
                            float *w = &channel_weights[ic * C4_K*C4_K];
                            for (int ky = 0; ky < C4_K; ky++) {
                                int iy = oy * C4_STRIDE + ky - PAD4_H;
                                for (int kx = 0; kx < C4_K; kx++) {
                                    int ix = ox * C4_STRIDE + kx - PAD4_W;
                                    if (iy >= 0 && iy < C3_H && ix >= 0 && ix < C3_W) {
                                        int in_idx = (ic * C3_H + iy) * C3_W + ix;
                                        sum += s->c3_out[in_idx] * w[ky*C4_K + kx];
                                    }
                                }
                            }
                        }
                        int out_idx = (oc * C4_H + oy) * C4_W + ox;
                        s->c4_out[out_idx] = relu_f(sum);
                    }
                }
            }
            s->worker_done[id] = 4;
        }

        if (expect == 5) {
            for (int j = 0; j < gap_cnt; j++) {
                int ch = gap_start + j;
                float sum = 0.0f;
                for (int y = 0; y < C4_H; ++y) {
                    for (int x = 0; x < C4_W; ++x) {
                        int idx = (ch * C4_H + y) * C4_W + x;
                        sum += s->c4_out[idx];
                    }
                }
                s->gap[ch] = sum / (C4_H * C4_W);
            }
            s->worker_done[id] = 5;
        }

        if (expect == 6) {
            for (int j = 0; j < dense_cnt; j++) {
                int o = dense_start + j;
                local_bias = dense_biases[o];
                for (int i = 0; i < GAP_OUT; i++) {
                    int idx = IDX_DENSE_W(i, o);
                    channel_weights[i] = dense_weights[idx];
                }
                float sum = local_bias;
                for (int i = 0; i < GAP_OUT; i++) {
                    sum += s->gap[i] * channel_weights[i];
                }
                s->out[o] = sum;
            }
            s->worker_done[id] = 6;
        }
    }
}

void main() {
    printf("\n=== Parallel MNIST CNN --- %s (%d workers) ===\n", MODEL_NAME, N_WORKERS);
    printf("Model size: %d parameters\n", TOTAL_PARAMS);
    printf("Sample index: %d, True label: %d\n\n", SAMPLE_INDEX, SAMPLE_LABEL);

    memset((void *)&NN, 0, sizeof(NN));
    NN.phase = 0;
    for (int i = 0; i < N_WORKERS; ++i) NN.worker_done[i] = 0;

    for (int y = 0; y < IN_H; ++y)
        for (int x = 0; x < IN_W; ++x)
            NN.img[y][x] = mnist_sample[y][x];

    static WorkerParam wparams[N_WORKERS];
    int cid_workers[N_WORKERS];
    for (int i = 0; i < N_WORKERS; ++i) {
        wparams[i].shared = &NN;
        wparams[i].id = i;
        cid_workers[i] = _cogstart_C(cog_worker, &wparams[i], stacks_workers[i], sizeof(stacks_workers[i]));
    }

    uint32_t t0 = _getus();

    for (int ph = 1; ph <= 6; ph++) {
        for (int i = 0; i < N_WORKERS; ++i) NN.worker_done[i] = 0;
        NN.phase = ph;
        int done;
        do {
            done = 1;
            for (int i = 0; i < N_WORKERS; ++i) if (NN.worker_done[i] != ph) { done = 0; break; }
            if (!done) _waitx(_clockfreq() / 
2000);
        } while (!done);
    }

    uint32_t t1 = _getus();
    uint32_t us_total = t1-t0;

    float logits[DENSE_OUT];
    for (int k = 0; k < DENSE_OUT; ++k) logits[k] = NN.out[k];
    softmax_f(logits, DENSE_OUT);

    int pred = 0;
    for (int i = 1; i < DENSE_OUT; ++i) if (logits[i] > logits[pred]) pred = i;

    printf("\nProbabilities:\n");
    for (int i = 0; i < DENSE_OUT; ++i) {
        printf("%d: %.6f", i, logits[i]);
        if (i == pred) printf("  <-- PREDICTION");
        if (i == SAMPLE_LABEL) printf("  (TRUE LABEL)");
        printf("\n");
    }
    printf("\nPredicted digit: %d\n", pred);
    printf("True digit: %d\n", SAMPLE_LABEL);
    printf("Correct: %s\n", (pred == SAMPLE_LABEL) ? "YES" : "NO");
    printf("Inference time: %lu.%01lu ms\n", us_total / 1000, us_total);

    for (int i = 0; i < N_WORKERS; ++i) _cogstop(cid_workers[i]);

    for (;;) _waitx(_clockfreq() / 2000);
}
