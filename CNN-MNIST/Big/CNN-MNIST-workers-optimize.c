//The weights here are quantized
//Run time: 3.71s
//No loss quality
//Weight: 370KB
//4 workers
//Optimized version of previous code
//For the calculations, the work is spread between for "workers" (cogs) in the sense that if 28 matrix columns are necessary, worker 1 will do 0-7, worker 2 will do 8-15, etc
//7 cogs are used, one for input, one for flags, one for main and 4 for calculations
//RELU and softmax have been used

#include <stdio.h>
#include <propeller2.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "mnist_cnn_weights_q.h"
#include "numero_matriz.h"

#define N_WORKERS 4
#define STACKSIZE   1024
unsigned char stack_input[STACKSIZE];
unsigned char stacks_workers[N_WORKERS][STACKSIZE];

#define IN_H   28
#define IN_W   28
#define IN_C    1

#define C1_OUT  32
#define C1_K     3
#define C1_H    (IN_H - (C1_K - 1))
#define C1_W    (IN_W - (C1_K - 1))
#define C1_P_H  (C1_H / 2)
#define C1_P_W  (C1_W / 2)

#define C2_OUT  64
#define C2_K     3
#define C2_H    (C1_P_H - (C2_K - 1))
#define C2_W    (C1_P_W - (C2_K - 1))
#define C2_P_H  (C2_H / 2)
#define C2_P_W  (C2_W / 2)

#define FC_IN   (C2_OUT * C2_P_H * C2_P_W)
#define FC1_OUT 128
#define FC2_OUT 10

#define VERY_NEGATIVE_FLOAT (-1e30f)

typedef struct {
    volatile float img2d[IN_H][IN_W];
    volatile float c1p[C1_OUT * C1_P_H * C1_P_W];
    volatile float c2p[C2_OUT * C2_P_H * C2_P_W];
    volatile float fc1[FC1_OUT];
    volatile float out[FC2_OUT];
    volatile int flag_input;
    volatile int phase;
    volatile int worker_done[N_WORKERS];
} SharedNN;

SharedNN NN;

typedef struct {
    SharedNN *shared;
    int id;
} WorkerParam;

static float conv1_wf[C1_OUT * IN_C * C1_K * C1_K];
static float conv2_wf[C2_OUT * C1_OUT * C2_K * C2_K];
static float fc2_wf[FC2_OUT * FC1_OUT];
static float conv1_bf[C1_OUT];
static float conv2_bf[C2_OUT];
static float fc1_bf[FC1_OUT];
static float fc2_bf[FC2_OUT];

static inline float relu_inline(float x) { return x < 0.0f ? 0.0f : x; }

static void softmax_f(float *x, int len) {
    float maxv = x[0];
    for (int i = 1; i < len; ++i) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) { x[i] = expf(x[i] - maxv); sum += x[i]; }
    for (int i = 0; i < len; ++i) x[i] /= sum;
}

static inline int idx_conv1_w(int out, int in, int kh, int kw) { return ((out * IN_C + in) * C1_K + kh) * C1_K + kw; }
static inline int idx_conv2_w(int out, int in, int kh, int kw) { return ((out * C1_OUT + in) * C2_K + kh) * C2_K + kw; }
static inline int idx_fc1_w(int out, int in) { return out * FC_IN + in; }
static inline int idx_fc2_w(int out, int in) { return out * FC1_OUT + in; }

void cog_input(void *p) {
    SharedNN *s = (SharedNN *)p;
    while (1) {
        if (s->flag_input == 2) {
            for (int y = 0; y < IN_H; ++y)
                for (int x = 0; x < IN_W; ++x)
                    s->img2d[y][x] = numero[y][x];
            s->flag_input = 1;
        }
        _waitx(100);
    }
}

void cog_worker(void *p) {
    WorkerParam *wp = (WorkerParam *)p;
    SharedNN *s = wp->shared;
    int id = wp->id;
    int c1_per = (C1_OUT + N_WORKERS - 1) / N_WORKERS;
    int c1_start = id * c1_per;
    int c1_end = c1_start + c1_per; if (c1_end > C1_OUT) c1_end = C1_OUT;
    int c2_per = (C2_OUT + N_WORKERS - 1) / N_WORKERS;
    int c2_start = id * c2_per;
    int c2_end = c2_start + c2_per; if (c2_end > C2_OUT) c2_end = C2_OUT;
    int fc1_per = (FC1_OUT + N_WORKERS - 1) / N_WORKERS;
    int fc1_start = id * fc1_per;
    int fc1_end = fc1_start + fc1_per; if (fc1_end > FC1_OUT) fc1_end = FC1_OUT;

    while (1) {
        int phase = s->phase;
        if (phase == 1) {
            for (int oc = c1_start; oc < c1_end; ++oc) {
                float bias = conv1_bf[oc];
                for (int oy = 0; oy < C1_P_H; ++oy) {
                    for (int ox = 0; ox < C1_P_W; ++ox) {
                        float maxv = VERY_NEGATIVE_FLOAT;
                        for (int py = 0; py < 2; ++py) {
                            for (int px = 0; px < 2; ++px) {
                                int conv_y = oy * 2 + py;
                                int conv_x = ox * 2 + px;
                                const float *wbase = &conv1_wf[idx_conv1_w(oc, 0, 0, 0)];
                                float ssum = bias
                                    + s->img2d[conv_y + 0][conv_x + 0] * wbase[0]
                                    + s->img2d[conv_y + 0][conv_x + 1] * wbase[1]
                                    + s->img2d[conv_y + 0][conv_x + 2] * wbase[2]
                                    + s->img2d[conv_y + 1][conv_x + 0] * wbase[3]
                                    + s->img2d[conv_y + 1][conv_x + 1] * wbase[4]
                                    + s->img2d[conv_y + 1][conv_x + 2] * wbase[5]
                                    + s->img2d[conv_y + 2][conv_x + 0] * wbase[6]
                                    + s->img2d[conv_y + 2][conv_x + 1] * wbase[7]
                                    + s->img2d[conv_y + 2][conv_x + 2] * wbase[8];
                                float aval = relu_inline(ssum);
                                if (aval > maxv) maxv = aval;
                            }
                        }
                        s->c1p[(oc * C1_P_H + oy) * C1_P_W + ox] = maxv;
                    }
                }
            }
            s->worker_done[id] = 1;
            while (s->phase == 1) _waitx(1);
            continue;
        }
        if (phase == 2) {
            for (int oc = c2_start; oc < c2_end; ++oc) {
                float bias = conv2_bf[oc];
                for (int oy = 0; oy < C2_P_H; ++oy) {
                    for (int ox = 0; ox < C2_P_W; ++ox) {
                        float maxv = VERY_NEGATIVE_FLOAT;
                        for (int py = 0; py < 2; ++py) {
                            for (int px = 0; px < 2; ++px) {
                                int conv_y = oy * 2 + py;
                                int conv_x = ox * 2 + px;
                                float ssum = bias;
                                for (int ic = 0; ic < C1_OUT; ++ic) {
                                    const float *wbase = &conv2_wf[idx_conv2_w(oc, ic, 0, 0)];
                                    const float *fmap = &s->c1p[(ic * C1_P_H + conv_y) * C1_P_W + conv_x];
                                    ssum += fmap[0] * wbase[0]
                                         + fmap[1] * wbase[1]
                                         + fmap[2] * wbase[2]
                                         + fmap[C1_P_W+0] * wbase[3]
                                         + fmap[C1_P_W+1] * wbase[4]
                                         + fmap[C1_P_W+2] * wbase[5]
                                         + fmap[2*C1_P_W+0] * wbase[6]
                                         + fmap[2*C1_P_W+1] * wbase[7]
                                         + fmap[2*C1_P_W+2] * wbase[8];
                                }
                                float aval = relu_inline(ssum);
                                if (aval > maxv) maxv = aval;
                            }
                        }
                        s->c2p[(oc * C2_P_H + oy) * C2_P_W + ox] = maxv;
                    }
                }
            }
            s->worker_done[id] = 2;
            while (s->phase == 2) _waitx(1);
            continue;
        }
        if (phase == 3) {
            for (int o = fc1_start; o < fc1_end; ++o) {
                float sum = fc1_bf[o];
                for (int i = 0; i < FC_IN; ++i) {
                    float w = (float)fc1_weight_q[idx_fc1_w(o, i)] * fc1_weight_scale;
                    sum += s->c2p[i] * w;
                }
                s->fc1[o] = relu_inline(sum);
            }
            s->worker_done[id] = 3;
            while (s->phase == 3) _waitx(1);
            continue;
        }
        _waitx(100);
    }
}

static void convert_weights_to_float(void) {
    int n_conv1 = C1_OUT * IN_C * C1_K * C1_K;
    int n_conv2 = C2_OUT * C1_OUT * C2_K * C2_K;
    int n_fc2   = FC2_OUT * FC1_OUT;
    for (int i = 0; i < n_conv1; ++i) conv1_wf[i] = (float)conv1_weight_q[i] * conv1_weight_scale;
    for (int i = 0; i < n_conv2; ++i) conv2_wf[i] = (float)conv2_weight_q[i] * conv2_weight_scale;
    for (int i = 0; i < n_fc2; ++i)   fc2_wf[i]   = (float)fc2_weight_q[i] * fc2_weight_scale;
    for (int i = 0; i < C1_OUT; ++i) conv1_bf[i] = conv1_bias ? conv1_bias[i] : 0.0f;
    for (int i = 0; i < C2_OUT; ++i) conv2_bf[i] = conv2_bias ? conv2_bias[i] : 0.0f;
    for (int i = 0; i < FC1_OUT; ++i) fc1_bf[i]   = fc1_bias ? fc1_bias[i] : 0.0f;
    for (int i = 0; i < FC2_OUT; ++i) fc2_bf[i]   = fc2_bias ? fc2_bias[i] : 0.0f;
}

int main() {
    printf("\n Optimizacion modelo trabajadores\n");
    memset((void *)&NN, 0, sizeof(NN));
    NN.flag_input = 0;
    NN.phase = 0;
    for (int i = 0; i < N_WORKERS; ++i) NN.worker_done[i] = 0;
    convert_weights_to_float();
    int cid_in = _cogstart_C(cog_input, &NN, stack_input, sizeof(stack_input));
    static WorkerParam wparams[N_WORKERS];
    int cid_workers[N_WORKERS];
    for (int i = 0; i < N_WORKERS; ++i) {
        wparams[i].shared = &NN;
        wparams[i].id = i;
        cid_workers[i] = _cogstart_C(cog_worker, &wparams[i], stacks_workers[i], sizeof(stacks_workers[i]));
    }
    NN.flag_input = 2;
    while (NN.flag_input != 1) _waitx(100);
    uint64_t ms_start = _getms();
    NN.phase = 1;
    for (int i = 0; i < N_WORKERS; ++i) NN.worker_done[i] = 0;
    int done;
    do {
        done = 1;
        for (int i = 0; i < N_WORKERS; ++i) if (NN.worker_done[i] != 1) { done = 0; break; }
        if (!done) _waitx(1);
    } while (!done);
    NN.phase = 0;
    NN.phase = 2;
    for (int i = 0; i < N_WORKERS; ++i) NN.worker_done[i] = 0;
    do {
        done = 1;
        for (int i = 0; i < N_WORKERS; ++i) if (NN.worker_done[i] != 2) { done = 0; break; }
        if (!done) _waitx(1);
    } while (!done);
    NN.phase = 0;
    NN.phase = 3;
    for (int i = 0; i < N_WORKERS; ++i) NN.worker_done[i] = 0;
    do {
        done = 1;
        for (int i = 0; i < N_WORKERS; ++i) if (NN.worker_done[i] != 3) { done = 0; break; }
        if (!done) _waitx(1);
    } while (!done);
    NN.phase = 0;
    for (int o = 0; o < FC2_OUT; ++o) {
        float sum = fc2_bf[o];
        const float *wbase = &fc2_wf[o * FC1_OUT];
        for (int i = 0; i < FC1_OUT; ++i) sum += NN.fc1[i] * wbase[i];
        NN.out[o] = sum;
    }
    uint64_t ms_end = _getms();
    float ms_total = (float)(ms_end - ms_start);
    float logits[FC2_OUT];
    for (int k = 0; k < FC2_OUT; ++k) logits[k] = NN.out[k];
    softmax_f(logits, FC2_OUT);
    int argmax = 0;
    for (int i = 1; i < FC2_OUT; ++i) if (logits[i] > logits[argmax]) argmax = i;
    printf("\nNum: %d\n", argmax);
    printf("Prob:\n");
    for (int i = 0; i < FC2_OUT; ++i) printf("  %d: %.4f\n", i, logits[i]);
    printf("Tiempo: %.3f ms\n", ms_total);
    _cogstop(cid_in);
    for (int i = 0; i < N_WORKERS; ++i) _cogstop(cid_workers[i]);
    for (;;) _waitx(1000);
    return 0;
}
