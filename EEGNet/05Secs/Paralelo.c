

enum { _clkfreq = 320000000 };

#include <stdio.h>
#include <propeller2.h>
#include <math.h>
#include <string.h>

#include "eegnet_weights_0p5s.h"
#include "sample_0p5s.h"

/* ------------------------------------------------------------------ */
#define N_CH     EEG_N_CHANNELS    /* 6   */
#define N_SAMP   EEG_N_SAMPLES     /* 125 */
#define N_CLS    EEG_N_CLASSES     /* 2   */
#define F1       EEG_F1            /* 8   */
#define F2       EEG_F2            /* 16  */
#define D        EEG_D             /* 2   */
#define K_LEN    EEG_KERNEL_LEN    /* 62  */
#define SEP_K    EEG_SEP_K         /* 16  */
#define POOL1_T  EEG_POOL1_T       /* 31  */
#define POOL2_T  EEG_POOL2_T       /* 3   */
#define FLAT_N   EEG_FLATTEN_SIZE  /* 48  */
#define BN_EPS   1e-3f

#define MAX_WORKERS  7
#define STACKSIZE    4096

/* ================================================================
   Shared hub-RAM structure
   ================================================================ */
typedef struct {
    volatile float input[N_CH][N_SAMP];
    volatile float c1  [N_CH ][N_SAMP ][F1];
    volatile float dw  [1    ][N_SAMP ][F2];
    volatile float p1  [1    ][POOL1_T][F2];
    volatile float sc  [1    ][POOL1_T][F2];
    volatile float p2  [1    ][POOL2_T][F2];
    volatile float flat[FLAT_N];
    volatile float out [N_CLS];

    volatile int phase;
    volatile int active_workers;
    volatile int worker_done [MAX_WORKERS];
    volatile int layer_start [MAX_WORKERS];
    volatile int layer_count [MAX_WORKERS];
} SharedNN;

static SharedNN NN;

/* ================================================================
   Phase descriptors: {nworkers, total_units}
   ================================================================ */
typedef struct { int nworkers; int total; } PhaseDesc;

static const PhaseDesc PHASES[10] = {
    {0, 0        },   /* 0 unused */
    {7, F1       },   /* 1  Conv2D temporal:      8  output channels */
    {7, F1       },   /* 2  BatchNorm 1:          8  channels        */
    {7, F2       },   /* 3  DepthwiseCconv:       16 output channels */
    {7, F2       },   /* 4  BatchNorm2 + ELU:     16 channels        */
    {7, POOL1_T  },   /* 5  AvgPool1:             31 time steps      */
    {7, F2       },   /* 6  SeparableConv:        16 output channels */
    {7, F2       },   /* 7  BatchNorm3 + ELU:     16 channels        */
    {3, POOL2_T  },   /* 8  AvgPool2:              3 time steps      */
    {1, N_CLS    },   /* 9  Dense + Softmax:       2 outputs         */
};

/* ================================================================
   Worker parameters
   ================================================================ */
typedef struct { SharedNN *nn; int id; } WorkerParam;

static WorkerParam   wparams[MAX_WORKERS];
static unsigned char wstacks[MAX_WORKERS][STACKSIZE];

/* ================================================================
   Helpers
   ================================================================ */
static inline float elu_f(float x)
{
    return (x >= 0.0f) ? x : (expf(x) - 1.0f);
}

static inline void get_worker_range(int total, int id, int nw,
                                    int *start, int *count)
{
    int base = total / nw;
    int rem  = total % nw;
    *start = id * base + (id < rem ? id : rem);
    *count = base + (id < rem ? 1 : 0);
}

/* ================================================================
   Per-worker layer implementations
   ================================================================ */

/* Phase 1: Conv2D temporal — slice over output channel oc */
static void ph1_conv2d(SharedNN *nn, int start, int count)
{
    int half = K_LEN / 2;
    for (int j = 0; j < count; j++) {
        int oc = start + j;
        for (int h = 0; h < N_CH; h++) {
            for (int t = 0; t < N_SAMP; t++) {
                float sum = 0.0f;
                for (int kx = 0; kx < K_LEN; kx++) {
                    int ti = t + kx - half;
                    if (ti >= 0 && ti < N_SAMP)
                        sum += nn->input[h][ti]
                             * conv2d_1_kernel[kx * F1 + oc];
                }
                nn->c1[h][t][oc] = sum;
            }
        }
    }
}

/* Phase 2: BatchNorm 1 — slice over channel k */
static void ph2_bn1(SharedNN *nn, int start, int count)
{
    for (int j = 0; j < count; j++) {
        int k  = start + j;
        float g  = bn_1_gamma[k];
        float b  = bn_1_beta[k];
        float m  = bn_1_moving_mean[k];
        float vi = 1.0f / sqrtf(bn_1_moving_variance[k] + BN_EPS);
        for (int h = 0; h < N_CH; h++)
            for (int t = 0; t < N_SAMP; t++)
                nn->c1[h][t][k] = g * (nn->c1[h][t][k] - m) * vi + b;
    }
}

/* Phase 3: DepthwiseCconv — slice over output channel oc = ic*D+dm */
static void ph3_depthwise(SharedNN *nn, int start, int count)
{
    for (int j = 0; j < count; j++) {
        int oc = start + j;
        int ic = oc / D;
        int dm = oc % D;
        for (int t = 0; t < N_SAMP; t++) {
            float sum = 0.0f;
            for (int h = 0; h < N_CH; h++) {
                int widx = h * F1 * D + ic * D + dm;
                sum += nn->c1[h][t][ic]
                     * depthwise_conv2d_1_kernel[widx];
            }
            nn->dw[0][t][oc] = sum;
        }
    }
}

/* Phase 4: BatchNorm2 + ELU — slice over channel k */
static void ph4_bn2_elu(SharedNN *nn, int start, int count)
{
    for (int j = 0; j < count; j++) {
        int k  = start + j;
        float g  = bn_2_gamma[k];
        float b  = bn_2_beta[k];
        float m  = bn_2_moving_mean[k];
        float vi = 1.0f / sqrtf(bn_2_moving_variance[k] + BN_EPS);
        for (int t = 0; t < N_SAMP; t++) {
            float v = g * (nn->dw[0][t][k] - m) * vi + b;
            nn->dw[0][t][k] = elu_f(v);
        }
    }
}

/* Phase 5: AvgPool1 (factor 4) — slice over output time step t */
static void ph5_avgpool1(SharedNN *nn, int start, int count)
{
    for (int j = 0; j < count; j++) {
        int t = start + j;
        for (int k = 0; k < F2; k++) {
            float s = nn->dw[0][t*4+0][k] + nn->dw[0][t*4+1][k]
                    + nn->dw[0][t*4+2][k] + nn->dw[0][t*4+3][k];
            nn->p1[0][t][k] = s * 0.25f;
        }
    }
}

/* Phase 6: SeparableConv2D — slice over output channel oc
   Depthwise + pointwise fused per output channel. */
static void ph6_sepconv(SharedNN *nn, int start, int count)
{
    int half = SEP_K / 2;
    for (int j = 0; j < count; j++) {
        int oc = start + j;
        for (int t = 0; t < POOL1_T; t++) {
            float psum = 0.0f;
            for (int ic = 0; ic < F2; ic++) {
                float dsum = 0.0f;
                for (int kx = 0; kx < SEP_K; kx++) {
                    int ti = t + kx - half;
                    if (ti >= 0 && ti < POOL1_T)
                        dsum += nn->p1[0][ti][ic]
                              * sep_dw_kernel[kx * F2 + ic];
                }
                psum += dsum * sep_pw_kernel[ic * F2 + oc];
            }
            nn->sc[0][t][oc] = psum;
        }
    }
}

/* Phase 7: BatchNorm3 + ELU — slice over channel k */
static void ph7_bn3_elu(SharedNN *nn, int start, int count)
{
    for (int j = 0; j < count; j++) {
        int k  = start + j;
        float g  = bn_3_gamma[k];
        float b  = bn_3_beta[k];
        float m  = bn_3_moving_mean[k];
        float vi = 1.0f / sqrtf(bn_3_moving_variance[k] + BN_EPS);
        for (int t = 0; t < POOL1_T; t++) {
            float v = g * (nn->sc[0][t][k] - m) * vi + b;
            nn->sc[0][t][k] = elu_f(v);
        }
    }
}

/* Phase 8: AvgPool2 (factor 8) — slice over output time step t
   Only 3 time steps total; workers 3-6 receive count=0 and do nothing. */
static void ph8_avgpool2(SharedNN *nn, int start, int count)
{
    for (int j = 0; j < count; j++) {
        int t = start + j;
        for (int k = 0; k < F2; k++) {
            float s = 0.0f;
            for (int p = 0; p < 8; p++)
                s += nn->sc[0][t * 8 + p][k];
            nn->p2[0][t][k] = s * 0.125f;
        }
    }
}

/* Phase 9: Dense + Softmax — worker 0 only */
static void ph9_dense_softmax(SharedNN *nn)
{
    int idx = 0;
    for (int t = 0; t < POOL2_T; t++)
        for (int k = 0; k < F2; k++)
            nn->flat[idx++] = nn->p2[0][t][k];

    float mx = -1e38f;
    for (int oc = 0; oc < N_CLS; oc++) {
        float s = dense_bias[oc];
        for (int i = 0; i < FLAT_N; i++)
            s += nn->flat[i] * dense_kernel[i * N_CLS + oc];
        nn->out[oc] = s;
        if (s > mx) mx = s;
    }
    float denom = 0.0f;
    for (int oc = 0; oc < N_CLS; oc++) {
        nn->out[oc] = expf(nn->out[oc] - mx);
        denom += nn->out[oc];
    }
    for (int oc = 0; oc < N_CLS; oc++)
        nn->out[oc] /= denom;
}

/* ================================================================
   Worker COG body — spins on NN.phase, executes slice, signals done
   ================================================================ */
static void cog_worker(void *p)
{
    WorkerParam *wp = (WorkerParam *)p;
    SharedNN    *nn = wp->nn;
    int          id = wp->id;

    while (1) {
        int ph = nn->phase;
        if (ph == -1) return;

        if (id >= nn->active_workers) {
            _waitx(_clockfreq() / 4000);
            continue;
        }

        int start = nn->layer_start[id];
        int count = nn->layer_count[id];

        if (count > 0) {
            switch (ph) {
                case 1: ph1_conv2d    (nn, start, count); break;
                case 2: ph2_bn1       (nn, start, count); break;
                case 3: ph3_depthwise (nn, start, count); break;
                case 4: ph4_bn2_elu   (nn, start, count); break;
                case 5: ph5_avgpool1  (nn, start, count); break;
                case 6: ph6_sepconv   (nn, start, count); break;
                case 7: ph7_bn3_elu   (nn, start, count); break;
                case 8: ph8_avgpool2  (nn, start, count); break;
                case 9: if (id == 0) ph9_dense_softmax(nn); break;
            }
        }

        nn->worker_done[id] = ph;

        while (nn->phase == ph)
            _waitx(_clockfreq() / 4000);
    }
}

/* ================================================================
   Supervisor: distribute slice ranges and wait for completion
   ================================================================ */
static void dispatch_phase(int ph)
{
    const PhaseDesc *pd = &PHASES[ph];
    int nw    = pd->nworkers;
    int total = pd->total;

    for (int i = 0; i < MAX_WORKERS; i++) {
        if (i < nw) {
            int s, c;
            get_worker_range(total, i, nw, &s, &c);
            NN.layer_start[i] = s;
            NN.layer_count[i] = c;
        } else {
            NN.layer_start[i] = 0;
            NN.layer_count[i] = 0;
        }
        NN.worker_done[i] = 0;
    }
    NN.active_workers = nw;
    NN.phase          = ph;

    int done;
    do {
        done = 1;
        for (int i = 0; i < nw; i++)
            if (NN.worker_done[i] != ph) { done = 0; break; }
        if (!done) _waitx(_clockfreq() / 4000);
    } while (!done);
}

/* ================================================================
   MAIN
   ================================================================ */
int main(void)
{
    printf("\n=== EEGNet Parallel Inference (Propeller 2, %d workers) ===\n",
           MAX_WORKERS);
    printf("Window  : 0.5 s  (%d samples @ 250 Hz)\n", N_SAMP);
    printf("Channels: %d  Classes: %d\n", N_CH, N_CLS);
    printf("Input   : subject=%s  session=%s  trial=%s\n",
           INPUT_SUBJECT, INPUT_SESSION, INPUT_TRIAL);

    memset((void *)&NN, 0, sizeof(NN));
    NN.phase = 0;

    for (int h = 0; h < N_CH; h++)
        for (int t = 0; t < N_SAMP; t++)
            NN.input[h][t] = input_sample[h][t];

    int cog_ids[MAX_WORKERS];
    for (int i = 0; i < MAX_WORKERS; i++) {
        wparams[i].nn = &NN;
        wparams[i].id = i;
        cog_ids[i] = _cogstart_C(cog_worker, &wparams[i],
                                  wstacks[i], STACKSIZE);
    }

    uint32_t t0 = _getus();
    for (int ph = 1; ph <= 9; ph++)
        dispatch_phase(ph);
    uint32_t us = _getus() - t0;

    int pred = 0;
    for (int i = 1; i < N_CLS; i++)
        if (NN.out[i] > NN.out[pred]) pred = i;

    printf("\nClass probabilities:\n");
    printf("  Class 0 (Left hand) : %.4f\n",  (double)NN.out[0]);
    printf("  Class 1 (Right hand): %.4f\n",  (double)NN.out[1]);
    printf("\nPredicted class : %d  (%s)\n",
           pred, pred == 0 ? "Left hand" : "Right hand");
    printf("True label      : %d\n", INPUT_LABEL);
    printf("Correct         : %s\n",
           pred == INPUT_LABEL ? "YES" : "NO");
    printf("Inference time  : %lu us  (%lu.%01lu ms)\n",
           (unsigned long)us,
           (unsigned long)(us / 1000),
           (unsigned long)(us % 1000));

    NN.phase = -1;
    _waitx(_clockfreq() / 100);
    for (int i = 0; i < MAX_WORKERS; i++)
        _cogstop(cog_ids[i]);

    for (;;) _waitx(_clockfreq() / 2000);
    return 0;
}
