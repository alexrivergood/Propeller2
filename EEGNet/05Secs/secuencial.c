/*
 * secuencial_0p5s.c
 * EEGNet 6-channel EEG inference — single-COG sequential execution
 * Window: 0.5 s  (125 samples @ 250 Hz)
 *
 * Architecture parameters
 * =======================
 *   N_CH     = 6       EEG channels
 *   N_SAMP   = 125     samples per trial
 *   F1       = 8       temporal conv filters
 *   F2       = 16      depthwise / sep output filters
 *   D        = 2       depth multiplier
 *   K_LEN    = 62      Conv2D temporal kernel  (= N_SAMP // 2)
 *   SEP_K    = 16      SepConv kernel          (= min(16, POOL1_T) = 16)
 *   POOL1_T  = 31      time steps after AvgPool1 (= N_SAMP // 4)
 *   POOL2_T  = 3       time steps after AvgPool2 (= POOL1_T // 8)
 *   FLAT_N   = 48      dense input size        (= POOL2_T * F2)
 *
 * Hub RAM budget (static globals)
 * ================================
 *   c1_buf  [6][125][8]   = 6000 floats =  23 KB
 *   dw_buf  [1][125][16]  = 2000 floats =   8 KB
 *   p1_buf  [1][31 ][16]  =  496 floats =   2 KB
 *   sc_buf  [1][31 ][16]  =  496 floats =   2 KB
 *   p2_buf  [1][3  ][16]  =   48 floats = 0.2 KB
 *   flat_buf[48]                          0.2 KB
 *   Total ≈ 35 KB — well within 512 KB hub RAM
 *
 * Compilation:
 *   flexspin --p2 -O2 -o eegnet_seq_0p5s.bin secuencial_0p5s.c
 *
 * Required headers in the same directory:
 *   eegnet_weights_0p5s.h   (from EEGNet_0p5s_train.py)
 *   sample_0p5s.h           (from export_sample_0p5s.py)
 */

enum { _clkfreq = 320000000 };

#include <stdio.h>
#include <propeller2.h>
#include <math.h>
#include <string.h>

#include "eegnet_weights_0p5s.h"
#include "sample_0p5s.h"   /* const float input_sample[N_CH][N_SAMP] */

/* ================================================================
   Tensor dimensions
   ================================================================ */
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

/* ================================================================
   Intermediate buffers (static hub RAM)
   ================================================================ */
static float c1_buf [N_CH  ][N_SAMP ][F1  ];
static float dw_buf [1     ][N_SAMP ][F2  ];
static float p1_buf [1     ][POOL1_T][F2  ];
static float sc_buf [1     ][POOL1_T][F2  ];
static float p2_buf [1     ][POOL2_T][F2  ];
static float flat_buf[FLAT_N];
static float out_buf [N_CLS];

/* ================================================================
   Activation helper
   ================================================================ */
static inline float elu_f(float x)
{
    return (x >= 0.0f) ? x : (expf(x) - 1.0f);
}

/* ================================================================
   Layer 1 – Conv2D temporal  (1 × K_LEN, same-pad, no bias)
   Input : input_sample[N_CH][N_SAMP]
   Output: c1_buf[N_CH][N_SAMP][F1]
   Kernel: conv2d_1_kernel[K_LEN * F1]   flat index kx*F1 + oc
   ================================================================ */
static void layer_conv2d_temporal(void)
{
    int half = K_LEN / 2;
    for (int h = 0; h < N_CH; h++) {
        for (int t = 0; t < N_SAMP; t++) {
            for (int oc = 0; oc < F1; oc++) {
                float sum = 0.0f;
                for (int kx = 0; kx < K_LEN; kx++) {
                    int ti = t + kx - half;
                    if (ti >= 0 && ti < N_SAMP)
                        sum += input_sample[h][ti]
                             * conv2d_1_kernel[kx * F1 + oc];
                }
                c1_buf[h][t][oc] = sum;
            }
        }
    }
}

/* ================================================================
   Batch Normalisation (in-place, over last/channel dimension)
   ================================================================ */
static void batchnorm_inplace(float *tensor, int rows, int cols, int ch,
                               const float *gamma, const float *beta,
                               const float *mm,    const float *mv)
{
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float *ptr = tensor + (r * cols + c) * ch;
            for (int k = 0; k < ch; k++) {
                ptr[k] = gamma[k] * (ptr[k] - mm[k])
                        / sqrtf(mv[k] + BN_EPS) + beta[k];
            }
        }
    }
}

/* ================================================================
   Layer 2 – BatchNorm on c1_buf  (N_CH × N_SAMP × F1)
   ================================================================ */
static void layer_bn1(void)
{
    batchnorm_inplace(&c1_buf[0][0][0], N_CH, N_SAMP, F1,
                      bn_1_gamma, bn_1_beta,
                      bn_1_moving_mean, bn_1_moving_variance);
}

/* ================================================================
   Layer 3 – DepthwiseConv2D  (N_CH × 1, depth_mult=D, valid-pad)
   Input : c1_buf[N_CH][N_SAMP][F1]
   Output: dw_buf[1][N_SAMP][F2]   F2 = F1 * D
   Kernel: depthwise_conv2d_1_kernel  flat index h*F1*D + ic*D + dm
   ================================================================ */
static void layer_depthwise(void)
{
    for (int t = 0; t < N_SAMP; t++) {
        for (int ic = 0; ic < F1; ic++) {
            for (int dm = 0; dm < D; dm++) {
                int oc = ic * D + dm;
                float sum = 0.0f;
                for (int h = 0; h < N_CH; h++) {
                    int widx = h * F1 * D + ic * D + dm;
                    sum += c1_buf[h][t][ic]
                         * depthwise_conv2d_1_kernel[widx];
                }
                dw_buf[0][t][oc] = sum;
            }
        }
    }
}

/* ================================================================
   Layer 4 – BatchNorm on dw_buf  (1 × N_SAMP × F2)
   Layer 5 – ELU in-place
   ================================================================ */
static void layer_bn2_elu(void)
{
    batchnorm_inplace(&dw_buf[0][0][0], 1, N_SAMP, F2,
                      bn_2_gamma, bn_2_beta,
                      bn_2_moving_mean, bn_2_moving_variance);
    for (int i = 0; i < N_SAMP * F2; i++)
        ((float *)dw_buf)[i] = elu_f(((float *)dw_buf)[i]);
}

/* ================================================================
   Layer 6 – AveragePooling2D (1 × 4)
   Input : dw_buf[1][N_SAMP=125][F2]
   Output: p1_buf[1][POOL1_T=31][F2]
   Note: 125 / 4 = 31.25 → floor = 31  (Keras default, no remainder used)
   ================================================================ */
static void layer_avgpool1(void)
{
    for (int t = 0; t < POOL1_T; t++) {
        for (int k = 0; k < F2; k++) {
            float s = 0.0f;
            for (int p = 0; p < 4; p++)
                s += dw_buf[0][t * 4 + p][k];
            p1_buf[0][t][k] = s * 0.25f;
        }
    }
}

/* ================================================================
   Layer 7 – SeparableConv2D (1 × SEP_K=16, F2 filters, same-pad, no bias)
   a) Depthwise  kernel: sep_dw_kernel[SEP_K][F2]      (depth_mult=1)
   b) Pointwise  kernel: sep_pw_kernel[F2][F2]
   Input/output: [1][POOL1_T=31][F2]
   ================================================================ */
static void layer_separable_conv2d(void)
{
    static float tmp[POOL1_T][F2];
    int half = SEP_K / 2;

    /* Depthwise pass */
    for (int t = 0; t < POOL1_T; t++) {
        for (int ic = 0; ic < F2; ic++) {
            float sum = 0.0f;
            for (int kx = 0; kx < SEP_K; kx++) {
                int ti = t + kx - half;
                if (ti >= 0 && ti < POOL1_T)
                    sum += p1_buf[0][ti][ic]
                         * sep_dw_kernel[kx * F2 + ic];
            }
            tmp[t][ic] = sum;
        }
    }

    /* Pointwise pass */
    for (int t = 0; t < POOL1_T; t++) {
        for (int oc = 0; oc < F2; oc++) {
            float sum = 0.0f;
            for (int ic = 0; ic < F2; ic++)
                sum += tmp[t][ic] * sep_pw_kernel[ic * F2 + oc];
            sc_buf[0][t][oc] = sum;
        }
    }
}

/* ================================================================
   Layer 8 – BatchNorm on sc_buf  (1 × POOL1_T × F2)
   Layer 9 – ELU in-place
   ================================================================ */
static void layer_bn3_elu(void)
{
    batchnorm_inplace(&sc_buf[0][0][0], 1, POOL1_T, F2,
                      bn_3_gamma, bn_3_beta,
                      bn_3_moving_mean, bn_3_moving_variance);
    for (int i = 0; i < POOL1_T * F2; i++)
        ((float *)sc_buf)[i] = elu_f(((float *)sc_buf)[i]);
}

/* ================================================================
   Layer 10 – AveragePooling2D (1 × 8)
   Input : sc_buf[1][POOL1_T=31][F2]
   Output: p2_buf[1][POOL2_T=3 ][F2]
   Note: 31 / 8 = 3.875 → floor = 3  (Keras default)
   ================================================================ */
static void layer_avgpool2(void)
{
    for (int t = 0; t < POOL2_T; t++) {
        for (int k = 0; k < F2; k++) {
            float s = 0.0f;
            for (int p = 0; p < 8; p++)
                s += sc_buf[0][t * 8 + p][k];
            p2_buf[0][t][k] = s * 0.125f;
        }
    }
}

/* ================================================================
   Layer 11 – Flatten
   ================================================================ */
static void layer_flatten(void)
{
    memcpy(flat_buf, &p2_buf[0][0][0], FLAT_N * sizeof(float));
}

/* ================================================================
   Layer 12 – Dense (48 → 2) + numerically-stable Softmax
   Weights: dense_kernel[FLAT_N][N_CLS], dense_bias[N_CLS]
   ================================================================ */
static void layer_dense_softmax(void)
{
    float mx = -1e38f;
    for (int oc = 0; oc < N_CLS; oc++) {
        float sum = dense_bias[oc];
        for (int i = 0; i < FLAT_N; i++)
            sum += flat_buf[i] * dense_kernel[i * N_CLS + oc];
        out_buf[oc] = sum;
        if (sum > mx) mx = sum;
    }
    float denom = 0.0f;
    for (int oc = 0; oc < N_CLS; oc++) {
        out_buf[oc] = expf(out_buf[oc] - mx);
        denom += out_buf[oc];
    }
    for (int oc = 0; oc < N_CLS; oc++)
        out_buf[oc] /= denom;
}

/* ================================================================
   Top-level inference — returns predicted class index
   ================================================================ */
int eegnet_infer(void)
{
    layer_conv2d_temporal();
    layer_bn1();
    layer_depthwise();
    layer_bn2_elu();
    layer_avgpool1();
    layer_separable_conv2d();
    layer_bn3_elu();
    layer_avgpool2();
    layer_flatten();
    layer_dense_softmax();

    int pred = 0;
    for (int i = 1; i < N_CLS; i++)
        if (out_buf[i] > out_buf[pred]) pred = i;
    return pred;
}

/* ================================================================
   MAIN
   ================================================================ */
int main(void)
{
    printf("\n=== EEGNet Sequential Inference (Propeller 2) ===\n");
    printf("Window  : 0.5 s  (%d samples @ 250 Hz)\n", N_SAMP);
    printf("Channels: %d  Classes: %d\n", N_CH, N_CLS);
    printf("Input   : subject=%s  session=%s  trial=%s\n",
           INPUT_SUBJECT, INPUT_SESSION, INPUT_TRIAL);

    uint32_t t0 = _getus();
    int pred = eegnet_infer();
    uint32_t us = _getus() - t0;

    printf("\nClass probabilities:\n");
    printf("  Class 0 (Left hand) : %.4f\n",  (double)out_buf[0]);
    printf("  Class 1 (Right hand): %.4f\n",  (double)out_buf[1]);
    printf("\nPredicted class : %d  (%s)\n",
           pred, pred == 0 ? "Left hand" : "Right hand");
    printf("True label      : %d\n", INPUT_LABEL);
    printf("Correct         : %s\n",
           pred == INPUT_LABEL ? "YES" : "NO");
    printf("Inference time  : %lu us  (%lu.%01lu ms)\n",
           (unsigned long)us,
           (unsigned long)(us / 1000),
           (unsigned long)(us % 1000));

    _pinh(63);
    _waitms(pred == INPUT_LABEL ? 100 : 500);
    _pinl(63);

    for (;;) _waitx(_clockfreq() / 2000);
    return 0;
}
