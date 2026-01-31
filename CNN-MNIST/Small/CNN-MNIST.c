//Inference: 2.4s
//Inference int16x16 multiplication: 0.5s

#include <propeller2.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "mnist_weights.h"
#include "mnist_sample.h"

#define IN_H 28
#define IN_W 28
#define IN_C 1
#define C1_OUT 4
#define C1_K 3
#define C1_H 28
#define C1_W 28
#define C2_OUT 8
#define C2_K 3
#define C2_H 14
#define C2_W 14
#define C2_STRIDE 2
#define C3_OUT 8
#define C3_K 3
#define C3_H 14
#define C3_W 14
#define C4_OUT 12
#define C4_K 3
#define C4_H 7
#define C4_W 7
#define C4_STRIDE 2
#define GAP_OUT 12
#define DENSE_OUT 10

static inline float relu(float x) { return x > 0 ? x : 0; }

void softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

void conv2d_same(const float* input, int in_channels, int in_h, int in_w,
                 float* output, int out_channels, int kernel_size, int stride,
                 const float* weights, const float* bias) {
    
    int out_h = (in_h + stride - 1) / stride;
    int out_w = (in_w + stride - 1) / stride;
    int pad_along_h = ((out_h - 1) * stride + kernel_size - in_h);
    int pad_along_w = ((out_w - 1) * stride + kernel_size - in_w);
    int pad_top = pad_along_h / 2;
    int pad_left = pad_along_w / 2;
    
    for (int i = 0; i < out_channels * out_h * out_w; i++) {
        output[i] = 0.0f;
    }
    
    for (int oc = 0; oc < out_channels; oc++) {
        float bias_val = bias[oc];
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = bias_val;
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int ky = 0; ky < kernel_size; ky++) {
                        for (int kx = 0; kx < kernel_size; kx++) {
                            int ih = oh * stride + ky - pad_top;
                            int iw = ow * stride + kx - pad_left;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                int in_idx = (ic * in_h + ih) * in_w + iw;
                                int w_idx = ((ky * kernel_size + kx) * in_channels + ic) * out_channels + oc;
                                sum += input[in_idx] * weights[w_idx];
                            }
                        }
                    }
                }
                int out_idx = (oc * out_h + oh) * out_w + ow;
                output[out_idx] = sum;
            }
        }
    }
}

void global_avg_pool(const float* in, int c, int h, int w, float* out) {
    float spatial_size = h * w;
    for (int ch = 0; ch < c; ch++) {
        float sum = 0.0f;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = (ch * h + y) * w + x;
                sum += in[idx];
            }
        }
        out[ch] = sum / spatial_size;
    }
}

void print_tensor(const char* name, const float* data, int c, int h, int w) {
    printf("%s (shape %dx%dx%d):\n", name, c, h, w);
    for (int ch = 0; ch < c && ch < 2; ch++) {
        printf("  Channel %d (first 5 values): ", ch);
        for (int i = 0; i < 5 && i < h*w; i++) {
            printf("%.6f ", data[ch*h*w + i]);
        }
        printf("\n");
    }
}

int main() {
    printf("\n=== MNIST Small CNN Inference (No BatchNorm) ===\n");
    printf("Model size: %d parameters (%.1f KB)\n", TOTAL_PARAMS, TOTAL_PARAMS * 4.0f / 1024);
    printf("Sample index: %d, True label: %d\n\n", SAMPLE_INDEX, SAMPLE_LABEL);
    
    float conv1_out[C1_OUT * C1_H * C1_W];
    float conv2_out[C2_OUT * C2_H * C2_W];
    float conv3_out[C3_OUT * C3_H * C3_W];
    float conv4_out[C4_OUT * C4_H * C4_W];
    float gap_out[GAP_OUT];
    float output[DENSE_OUT];
    
    uint64_t t0 = _getcnt();
    
    conv2d_same((float*)mnist_sample, IN_C, IN_H, IN_W,
                conv1_out, C1_OUT, C1_K, 1,
                conv2d_weights, conv2d_biases);
    
    print_tensor("Conv1 output", conv1_out, C1_OUT, C1_H, C1_W);
    
    for (int i = 0; i < C1_OUT * C1_H * C1_W; i++) {
        conv1_out[i] = relu(conv1_out[i]);
    }
    
    print_tensor("Conv1 after ReLU", conv1_out, C1_OUT, C1_H, C1_W);
    
    conv2d_same(conv1_out, C1_OUT, C1_H, C1_W,
                conv2_out, C2_OUT, C2_K, C2_STRIDE,
                conv2d_1_weights, conv2d_1_biases);
    
    print_tensor("Conv2 output", conv2_out, C2_OUT, C2_H, C2_W);
    
    for (int i = 0; i < C2_OUT * C2_H * C2_W; i++) {
        conv2_out[i] = relu(conv2_out[i]);
    }
    
    print_tensor("Conv2 after ReLU", conv2_out, C2_OUT, C2_H, C2_W);
    
    conv2d_same(conv2_out, C2_OUT, C2_H, C2_W,
                conv3_out, C3_OUT, C3_K, 1,
                conv2d_2_weights, conv2d_2_biases);
    
    for (int i = 0; i < C3_OUT * C3_H * C3_W; i++) {
        conv3_out[i] = relu(conv3_out[i]);
    }
    
    float conv3_min = 1e9, conv3_max = -1e9, conv3_sum = 0;
    for (int i = 0; i < C3_OUT * C3_H * C3_W; i++) {
        float val = conv3_out[i];
        if (val < conv3_min) conv3_min = val;
        if (val > conv3_max) conv3_max = val;
        conv3_sum += val;
    }
    printf("\nConv3 stats: min=%.6f, max=%.6f, avg=%.6f\n", 
           conv3_min, conv3_max, conv3_sum/(C3_OUT * C3_H * C3_W));
    
    conv2d_same(conv3_out, C3_OUT, C3_H, C3_W,
                conv4_out, C4_OUT, C4_K, C4_STRIDE,
                conv2d_3_weights, conv2d_3_biases);
    
    for (int i = 0; i < C4_OUT * C4_H * C4_W; i++) {
        conv4_out[i] = relu(conv4_out[i]);
    }
    
    float conv4_min = 1e9, conv4_max = -1e9, conv4_sum = 0;
    for (int i = 0; i < C4_OUT * C4_H * C4_W; i++) {
        float val = conv4_out[i];
        if (val < conv4_min) conv4_min = val;
        if (val > conv4_max) conv4_max = val;
        conv4_sum += val;
    }
    printf("\nConv4 stats: min=%.6f, max=%.6f, avg=%.6f\n", 
           conv4_min, conv4_max, conv4_sum/(C4_OUT * C4_H * C4_W));
    
    print_tensor("Conv4 output", conv4_out, C4_OUT, C4_H, C4_W);
    
    global_avg_pool(conv4_out, C4_OUT, C4_H, C4_W, gap_out);
    
    printf("\nGAP outputs (12 features):\n");
    for (int i = 0; i < GAP_OUT; i++) {
        printf("  [%d]: %.6f\n", i, gap_out[i]);
    }
    
    printf("\nDense layer weights (first 3x3):\n");
    for (int i = 0; i < 3; i++) {
        printf("  ");
        for (int j = 0; j < 3; j++) {
            int idx = i * DENSE_OUT + j;
            printf("W[%d][%d]=%.6f ", i, j, dense_weights[idx]);
        }
        printf("\n");
    }
    
    printf("\nDense layer biases (all 10):\n");
    for (int o = 0; o < DENSE_OUT; o++) {
        printf("  b[%d]=%.6f\n", o, dense_biases[o]);
    }
    
    for (int o = 0; o < DENSE_OUT; o++) {
        float sum = dense_biases[o];
        for (int i = 0; i < GAP_OUT; i++) {
            int weight_idx = i * DENSE_OUT + o;
            sum += gap_out[i] * dense_weights[weight_idx];
        }
        output[o] = sum;
    }
    
    printf("\nLogits before softmax:\n");
    for (int i = 0; i < DENSE_OUT; i++) {
        printf("  [%d]: %.6f\n", i, output[i]);
    }
    
    softmax(output, DENSE_OUT);
    
    uint64_t t1 = _getcnt();
    float time_ms = (float)(t1 - t0) * 1000.0f / (float)_clockfreq();
    
    int pred = 0;
    float max_prob = output[0];
    for (int i = 1; i < DENSE_OUT; i++) {
        if (output[i] > max_prob) {
            pred = i;
            max_prob = output[i];
        }
    }
    
    printf("\nProbabilities:\n");
    for (int i = 0; i < DENSE_OUT; i++) {
        printf("%d: %.6f", i, output[i]);
        if (i == pred) printf("  <-- PREDICTION");
        if (i == SAMPLE_LABEL) printf("  (TRUE LABEL)");
        printf("\n");
    }
    
    printf("\nPredicted digit: %d (confidence: %.2f%%)\n", pred, max_prob*100);
    printf("True digit: %d\n", SAMPLE_LABEL);
    printf("Correct: %s\n", (pred == SAMPLE_LABEL) ? "YES" : "NO");
    printf("Inference time: %.3f ms\n", time_ms);
    
    printf("\nSample image statistics:\n");
    float img_min = 1e9, img_max = -1e9, img_sum = 0;
    int non_zero = 0;
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            float val = mnist_sample[y][x];
            if (val < img_min) img_min = val;
            if (val > img_max) img_max = val;
            img_sum += val;
            if (val > 0.1) non_zero++;
        }
    }
    printf("  Min: %.3f, Max: %.3f, Avg: %.3f\n", img_min, img_max, img_sum/(28*28));
    printf("  Active pixels (>0.1): %d/%d\n", non_zero, 28*28);
    
    printf("\nMemory usage summary:\n");
    printf("  conv1_out: %d floats = %d bytes\n", C1_OUT*C1_H*C1_W, C1_OUT*C1_H*C1_W*4);
    printf("  conv2_out: %d floats = %d bytes\n", C2_OUT*C2_H*C2_W, C2_OUT*C2_H*C2_W*4);
    printf("  conv3_out: %d floats = %d bytes\n", C3_OUT*C3_H*C3_W, C3_OUT*C3_H*C3_W*4);
    printf("  conv4_out: %d floats = %d bytes\n", C4_OUT*C4_H*C4_W, C4_OUT*C4_H*C4_W*4);
    printf("  Total intermediate storage: %d bytes\n", 
           (C1_OUT*C1_H*C1_W + C2_OUT*C2_H*C2_W + C3_OUT*C3_H*C3_W + C4_OUT*C4_H*C4_W)*4);
    
    return 0;
}
