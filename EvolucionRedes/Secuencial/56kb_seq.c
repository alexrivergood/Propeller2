#include <propeller2.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "56kb_weights.h"
#include "mnist_sample.h"

#define IN_H   28
#define IN_W   28
#define IN_C    1
#define C1_OUT  6
#define C1_K    3
#define C1_H    28
#define C1_W    28
#define C1_STRIDE 1
#define C2_OUT  12
#define C2_K    3
#define C2_H    14
#define C2_W    14
#define C2_STRIDE 2
#define C3_OUT  12
#define C3_K    3
#define C3_H    14
#define C3_W    14
#define C3_STRIDE 1
#define C4_OUT  12
#define C4_K    3
#define C4_H    7
#define C4_W    7
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

static inline float relu(float x) { return x > 0 ? x : 0; }

void softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static void conv2d_same_opt(const float* input, int in_c, int in_h, int in_w,
                            float* output, int out_c, int k, int stride,
                            const float* weights, const float* bias,
                            int pad_h, int pad_w) {
    int out_h = (in_h + stride - 1) / stride;
    int out_w = (in_w + stride - 1) / stride;

    __attribute__((cog)) float w_buf[108];

    for (int oc = 0; oc < out_c; ++oc) {
        float bias_val = bias[oc];
        for (int ic = 0; ic < in_c; ++ic) {
            for (int ky = 0; ky < k; ++ky) {
                for (int kx = 0; kx < k; ++kx) {
                    int idx = ((ky * k + kx) * in_c + ic) * out_c + oc;
                    w_buf[ic * k * k + ky * k + kx] = weights[idx];
                }
            }
        }
        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox = 0; ox < out_w; ++ox) {
                float sum = bias_val;
                for (int ic = 0; ic < in_c; ++ic) {
                    for (int ky = 0; ky < k; ++ky) {
                        int iy = oy * stride + ky - pad_h;
                        for (int kx = 0; kx < k; ++kx) {
                            int ix = ox * stride + kx - pad_w;
                            if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w) {
                                int in_idx = (ic * in_h + iy) * in_w + ix;
                                sum += input[in_idx] * w_buf[ic * k * k + ky * k + kx];
                            }
                        }
                    }
                }
                int out_idx = (oc * out_h + oy) * out_w + ox;
                output[out_idx] = relu(sum);
            }
        }
    }
}

void global_avg_pool(const float* in, int c, int h, int w, float* out) {
    float spatial = h * w;
    for (int ch = 0; ch < c; ++ch) {
        float sum = 0.0f;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                sum += in[(ch * h + y) * w + x];
        out[ch] = sum / spatial;
    }
}

void dense_layer(const float* in, int in_size, float* out, int out_size,
                 const float* weights, const float* bias) {
    for (int o = 0; o < out_size; ++o) {
        float sum = bias[o];
        for (int i = 0; i < in_size; ++i)
            sum += in[i] * weights[i * out_size + o];
        out[o] = sum;
    }
}

void print_tensor(const char* name, const float* data, int c, int h, int w) {
    printf("%s (shape %dx%dx%d):\n", name, c, h, w);
    for (int ch = 0; ch < c && ch < 2; ++ch) {
        printf("  Channel %d (first 5 values): ", ch);
        for (int i = 0; i < 5 && i < h*w; ++i)
            printf("%.6f ", data[ch*h*w + i]);
        printf("\n");
    }
}

int main() {
    printf("\n=== MNIST CNN Inference --- 56kb (Single-Cog) ===\n");
    printf("Model size: %d parameters (%.1f KB)\n", TOTAL_PARAMS, TOTAL_PARAMS * 4.0f / 1024);
    printf("Sample index: %d, True label: %d\n\n", SAMPLE_INDEX, SAMPLE_LABEL);

    static float conv1_out[C1_OUT * C1_H * C1_W];
    static float conv2_out[C2_OUT * C2_H * C2_W];
    static float conv3_out[C3_OUT * C3_H * C3_W];
    static float conv4_out[C4_OUT * C4_H * C4_W];
    static float gap_out[GAP_OUT];
    static float output[DENSE_OUT];

    uint32_t t0 = _getcnt();

    conv2d_same_opt((float*)mnist_sample, IN_C, IN_H, IN_W,
                    conv1_out, C1_OUT, C1_K, C1_STRIDE,
                    conv2d_weights, conv2d_biases, PAD1_H, PAD1_W);
    print_tensor("Conv1 after ReLU", conv1_out, C1_OUT, C1_H, C1_W);

    conv2d_same_opt(conv1_out, C1_OUT, C1_H, C1_W,
                    conv2_out, C2_OUT, C2_K, C2_STRIDE,
                    conv2d_1_weights, conv2d_1_biases, PAD2_H, PAD2_W);
    print_tensor("Conv2 after ReLU", conv2_out, C2_OUT, C2_H, C2_W);

    conv2d_same_opt(conv2_out, C2_OUT, C2_H, C2_W,
                    conv3_out, C3_OUT, C3_K, C3_STRIDE,
                    conv2d_2_weights, conv2d_2_biases, PAD3_H, PAD3_W);
    float conv3_min = 1e9, conv3_max = -1e9, conv3_sum = 0;
    for (int i = 0; i < C3_OUT * C3_H * C3_W; ++i) {
        float v = conv3_out[i];
        if (v < conv3_min) conv3_min = v;
        if (v > conv3_max) conv3_max = v;
        conv3_sum += v;
    }
    printf("\nConv3 stats: min=%.6f, max=%.6f, avg=%.6f\n",
           conv3_min, conv3_max, conv3_sum / (C3_OUT * C3_H * C3_W));

    conv2d_same_opt(conv3_out, C3_OUT, C3_H, C3_W,
                    conv4_out, C4_OUT, C4_K, C4_STRIDE,
                    conv2d_3_weights, conv2d_3_biases, PAD4_H, PAD4_W);
    print_tensor("Conv4 output", conv4_out, C4_OUT, C4_H, C4_W);
    float conv4_min = 1e9, conv4_max = -1e9, conv4_sum = 0;
    for (int i = 0; i < C4_OUT * C4_H * C4_W; ++i) {
        float v = conv4_out[i];
        if (v < conv4_min) conv4_min = v;
        if (v > conv4_max) conv4_max = v;
        conv4_sum += v;
    }
    printf("\nConv4 stats: min=%.6f, max=%.6f, avg=%.6f\n",
           conv4_min, conv4_max, conv4_sum / (C4_OUT * C4_H * C4_W));

    global_avg_pool(conv4_out, C4_OUT, C4_H, C4_W, gap_out);
    printf("\nGAP outputs (%d features):\n", GAP_OUT);
    for (int i = 0; i < GAP_OUT; ++i)
        printf("  [%d]: %.6f\n", i, gap_out[i]);

    dense_layer(gap_out, GAP_OUT, output, DENSE_OUT, dense_weights, dense_biases);

    printf("\nLogits before softmax:\n");
    for (int i = 0; i < DENSE_OUT; ++i)
        printf("  [%d]: %.6f\n", i, output[i]);

    softmax(output, DENSE_OUT);

    uint32_t t1 = _getcnt();
    float time_ms = (float)(t1 - t0) / ((float)_clockfreq() / 1000.0f);

    int pred = 0;
    float maxp = output[0];
    for (int i = 1; i < DENSE_OUT; ++i) if (output[i] > maxp) { maxp = output[i]; pred = i; }

    printf("\nProbabilities:\n");
    for (int i = 0; i < DENSE_OUT; ++i) {
        printf("%d: %.6f", i, output[i]);
        if (i == pred) printf("  <-- PREDICTION");
        if (i == SAMPLE_LABEL) printf("  (TRUE LABEL)");
        printf("\n");
    }
    printf("\nPredicted digit: %d (confidence: %.2f%%)\n", pred, maxp * 100);
    printf("True digit: %d\n", SAMPLE_LABEL);
    printf("Correct: %s\n", (pred == SAMPLE_LABEL) ? "YES" : "NO");
    printf("Inference time: %.3f ms\n", time_ms);

    for (;;) _waitx(_clockfreq() / 2000);
    return 0;
}
