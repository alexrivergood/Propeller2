#include <propeller2.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "mnist_weights_int8.h"  // Changed from mnist_weights.h
#include "mnist_sample.h"

// Worker configuration
#define N_WORKERS 4
#define STACKSIZE 1024
unsigned char stacks_workers[N_WORKERS][STACKSIZE];
unsigned char stack_main[STACKSIZE];

// Model architecture constants
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

// Shared data structure for inter-cog communication
typedef struct {
    // Input image (copied by input cog)
    volatile float img[IN_H][IN_W];
    
    // Intermediate feature maps (flat arrays for better access)
    volatile float c1_out[C1_OUT * C1_H * C1_W];
    volatile float c2_out[C2_OUT * C2_H * C2_W];
    volatile float c3_out[C3_OUT * C3_H * C3_W];
    volatile float c4_out[C4_OUT * C4_H * C4_W];
    
    // Global average pooling output
    volatile float gap[GAP_OUT];
    
    // Final output
    volatile float out[DENSE_OUT];
    
    // Control flags
    volatile int flag_input;
    volatile int phase;
    volatile int worker_done[N_WORKERS];
} SharedNN;

SharedNN NN;

// Worker parameters
typedef struct {
    SharedNN *shared;
    int id;
} WorkerParam;

static inline float relu_inline(float x) { return x < 0.0f ? 0.0f : x; }

static void softmax_f(float *x, int len) {
    float maxv = x[0];
    for (int i = 1; i < len; ++i) 
        if (x[i] > maxv) maxv = x[i];
    
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) { 
        x[i] = expf(x[i] - maxv); 
        sum += x[i]; 
    }
    
    for (int i = 0; i < len; ++i) 
        x[i] /= sum;
}

// Input cog - copies input image to shared memory
void cog_input(void *p) {
    SharedNN *s = (SharedNN *)p;
    while (1) {
        if (s->flag_input == 2) {
            // Copy input image from mnist_sample to shared memory
            for (int y = 0; y < IN_H; ++y)
                for (int x = 0; x < IN_W; ++x)
                    s->img[y][x] = mnist_sample[y][x];
            s->flag_input = 1;
        }
        _waitx(100);
    }
}

// Helper function to compute index for flat arrays
static inline int idx_3d(int ch, int h, int w, int H, int W) {
    return (ch * H + h) * W + w;
}

// Dequantize int8 weight: weight_int8 * scale
static inline float dequantize_int8(int8_t weight, float scale) {
    return (float)weight * scale;
}

// Worker cog - performs parallel computation with INT8 weights
void cog_worker(void *p) {
    WorkerParam *wp = (WorkerParam *)p;
    SharedNN *s = wp->shared;
    int id = wp->id;
    
    // Calculate workload distribution for each layer
    int c1_per = (C1_OUT + N_WORKERS - 1) / N_WORKERS;
    int c1_start = id * c1_per;
    int c1_end = c1_start + c1_per; 
    if (c1_end > C1_OUT) c1_end = C1_OUT;
    
    int c2_per = (C2_OUT + N_WORKERS - 1) / N_WORKERS;
    int c2_start = id * c2_per;
    int c2_end = c2_start + c2_per; 
    if (c2_end > C2_OUT) c2_end = C2_OUT;
    
    int c3_per = (C3_OUT + N_WORKERS - 1) / N_WORKERS;
    int c3_start = id * c3_per;
    int c3_end = c3_start + c3_per; 
    if (c3_end > C3_OUT) c3_end = C3_OUT;
    
    int c4_per = (C4_OUT + N_WORKERS - 1) / N_WORKERS;
    int c4_start = id * c4_per;
    int c4_end = c4_start + c4_per; 
    if (c4_end > C4_OUT) c4_end = C4_OUT;
    
    int gap_per = (GAP_OUT + N_WORKERS - 1) / N_WORKERS;
    int gap_start = id * gap_per;
    int gap_end = gap_start + gap_per;
    if (gap_end > GAP_OUT) gap_end = GAP_OUT;
    
    while (1) {
        int phase = s->phase;
        
        // ----- PHASE 1: Conv1 + ReLU (with INT8 weights) -----
        if (phase == 1) {
            // Each worker computes its assigned output channels for Conv1
            for (int oc = c1_start; oc < c1_end; ++oc) {
                float bias = conv2d_biases[oc];
                
                for (int oy = 0; oy < C1_H; ++oy) {
                    for (int ox = 0; ox < C1_W; ++ox) {
                        float sum = bias;
                        
                        // 3x3 convolution with same padding
                        for (int ky = 0; ky < 3; ++ky) {
                            for (int kx = 0; kx < 3; ++kx) {
                                int iy = oy + ky - 1;  // Same padding: pad=1
                                int ix = ox + kx - 1;
                                
                                if (iy >= 0 && iy < IN_H && ix >= 0 && ix < IN_W) {
                                    // TensorFlow order: [kH, kW, inC, outC]
                                    int w_idx = ((ky * 3 + kx) * IN_C) * C1_OUT + oc;
                                    // Dequantize INT8 weight
                                    float weight = dequantize_int8(conv2d_weights[w_idx], conv2d_scale);
                                    sum += s->img[iy][ix] * weight;
                                }
                            }
                        }
                        
                        int out_idx = idx_3d(oc, oy, ox, C1_H, C1_W);
                        s->c1_out[out_idx] = relu_inline(sum);
                    }
                }
            }
            s->worker_done[id] = 1;
            while (s->phase == 1) _waitx(1);
            continue;
        }
        
        // ----- PHASE 2: Conv2 + ReLU (with INT8 weights) -----
        if (phase == 2) {
            // Each worker computes its assigned output channels for Conv2
            for (int oc = c2_start; oc < c2_end; ++oc) {
                float bias = conv2d_1_biases[oc];
                
                for (int oy = 0; oy < C2_H; ++oy) {
                    for (int ox = 0; ox < C2_W; ++ox) {
                        float sum = bias;
                        
                        // 3x3 convolution with stride 2 and same padding
                        for (int ky = 0; ky < 3; ++ky) {
                            for (int kx = 0; kx < 3; ++kx) {
                                int iy = oy * 2 + ky - 0;  // Stride 2, same padding: pad=0
                                int ix = ox * 2 + kx - 0;
                                
                                if (iy >= 0 && iy < C1_H && ix >= 0 && ix < C1_W) {
                                    for (int ic = 0; ic < C1_OUT; ++ic) {
                                        int in_idx = idx_3d(ic, iy, ix, C1_H, C1_W);
                                        // TensorFlow order: [kH, kW, inC, outC]
                                        int w_idx = (((ky * 3 + kx) * C1_OUT + ic) * C2_OUT) + oc;
                                        // Dequantize INT8 weight
                                        float weight = dequantize_int8(conv2d_1_weights[w_idx], conv2d_1_scale);
                                        sum += s->c1_out[in_idx] * weight;
                                    }
                                }
                            }
                        }
                        
                        int out_idx = idx_3d(oc, oy, ox, C2_H, C2_W);
                        s->c2_out[out_idx] = relu_inline(sum);
                    }
                }
            }
            s->worker_done[id] = 2;
            while (s->phase == 2) _waitx(1);
            continue;
        }
        
        // ----- PHASE 3: Conv3 + ReLU (with INT8 weights) -----
        if (phase == 3) {
            // Each worker computes its assigned output channels for Conv3
            for (int oc = c3_start; oc < c3_end; ++oc) {
                float bias = conv2d_2_biases[oc];
                
                for (int oy = 0; oy < C3_H; ++oy) {
                    for (int ox = 0; ox < C3_W; ++ox) {
                        float sum = bias;
                        
                        // 3x3 convolution with stride 1 and same padding
                        for (int ky = 0; ky < 3; ++ky) {
                            for (int kx = 0; kx < 3; ++kx) {
                                int iy = oy + ky - 1;  // Same padding: pad=1
                                int ix = ox + kx - 1;
                                
                                if (iy >= 0 && iy < C2_H && ix >= 0 && ix < C2_W) {
                                    for (int ic = 0; ic < C2_OUT; ++ic) {
                                        int in_idx = idx_3d(ic, iy, ix, C2_H, C2_W);
                                        // TensorFlow order: [kH, kW, inC, outC]
                                        int w_idx = (((ky * 3 + kx) * C2_OUT + ic) * C3_OUT) + oc;
                                        // Dequantize INT8 weight
                                        float weight = dequantize_int8(conv2d_2_weights[w_idx], conv2d_2_scale);
                                        sum += s->c2_out[in_idx] * weight;
                                    }
                                }
                            }
                        }
                        
                        int out_idx = idx_3d(oc, oy, ox, C3_H, C3_W);
                        s->c3_out[out_idx] = relu_inline(sum);
                    }
                }
            }
            s->worker_done[id] = 3;
            while (s->phase == 3) _waitx(1);
            continue;
        }
        
        // ----- PHASE 4: Conv4 + ReLU (with INT8 weights) -----
        if (phase == 4) {
            // Each worker computes its assigned output channels for Conv4
            for (int oc = c4_start; oc < c4_end; ++oc) {
                float bias = conv2d_3_biases[oc];
                
                for (int oy = 0; oy < C4_H; ++oy) {
                    for (int ox = 0; ox < C4_W; ++ox) {
                        float sum = bias;
                        
                        // 3x3 convolution with stride 2 and same padding
                        for (int ky = 0; ky < 3; ++ky) {
                            for (int kx = 0; kx < 3; ++kx) {
                                int iy = oy * 2 + ky - 0;  // Stride 2, same padding: pad=0
                                int ix = ox * 2 + kx - 0;
                                
                                if (iy >= 0 && iy < C3_H && ix >= 0 && ix < C3_W) {
                                    for (int ic = 0; ic < C3_OUT; ++ic) {
                                        int in_idx = idx_3d(ic, iy, ix, C3_H, C3_W);
                                        // TensorFlow order: [kH, kW, inC, outC]
                                        int w_idx = (((ky * 3 + kx) * C3_OUT + ic) * C4_OUT) + oc;
                                        // Dequantize INT8 weight
                                        float weight = dequantize_int8(conv2d_3_weights[w_idx], conv2d_3_scale);
                                        sum += s->c3_out[in_idx] * weight;
                                    }
                                }
                            }
                        }
                        
                        int out_idx = idx_3d(oc, oy, ox, C4_H, C4_W);
                        s->c4_out[out_idx] = relu_inline(sum);
                    }
                }
            }
            s->worker_done[id] = 4;
            while (s->phase == 4) _waitx(1);
            continue;
        }
        
        // ----- PHASE 5: Global Average Pooling -----
        if (phase == 5) {
            // Each worker computes GAP for its assigned channels
            for (int ch = gap_start; ch < gap_end; ++ch) {
                float sum = 0.0f;
                for (int y = 0; y < C4_H; ++y) {
                    for (int x = 0; x < C4_W; ++x) {
                        int idx = idx_3d(ch, y, x, C4_H, C4_W);
                        sum += s->c4_out[idx];
                    }
                }
                s->gap[ch] = sum / (C4_H * C4_W);
            }
            s->worker_done[id] = 5;
            while (s->phase == 5) _waitx(1);
            continue;
        }
        
        _waitx(100);
    }
}

int main() {
    printf("\n=== MNIST Small CNN with Multi-Cog Parallelization ===\n");
    printf("Model size: %d parameters\n", TOTAL_PARAMS);
    printf("Original size (FP32): %.1f KB\n", ORIGINAL_SIZE_KB);
    printf("Quantized size (INT8): %.1f KB\n", QUANTIZED_SIZE_KB);
    printf("Compression ratio: %.1fx\n", COMPRESSION_RATIO);
    printf("Quantization: INT%d-bit\n", QUANTIZATION_BITS);
    printf("Using %d worker cogs for parallel computation\n", N_WORKERS);
    printf("Sample index: %d, True label: %d\n\n", SAMPLE_INDEX, SAMPLE_LABEL);
    
    // Initialize shared structure
    memset((void *)&NN, 0, sizeof(NN));
    NN.flag_input = 0;
    NN.phase = 0;
    for (int i = 0; i < N_WORKERS; ++i) 
        NN.worker_done[i] = 0;
    
    // Start input cog
    int cid_input = _cogstart_C(cog_input, &NN, stack_main, sizeof(stack_main));
    
    // Start worker cogs
    WorkerParam wparams[N_WORKERS];
    int cid_workers[N_WORKERS];
    for (int i = 0; i < N_WORKERS; ++i) {
        wparams[i].shared = &NN;
        wparams[i].id = i;
        cid_workers[i] = _cogstart_C(cog_worker, &wparams[i], stacks_workers[i], sizeof(stacks_workers[i]));
    }
    
    // Load input image
    NN.flag_input = 2;
    while (NN.flag_input != 1) _waitx(100);
    
    uint64_t ms_start = _getms();
    
    // ----- PHASE 1: Conv1 -----
    NN.phase = 1;
    for (int i = 0; i < N_WORKERS; ++i) NN.worker_done[i] = 0;
    int done;
    do {
        done = 1;
        for (int i = 0; i < N_WORKERS; ++i) 
            if (NN.worker_done[i] != 1) { 
                done = 0; 
                break; 
            }
        if (!done) _waitx(1);
    } while (!done);
    NN.phase = 0;
    
    // ----- PHASE 2: Conv2 -----
    NN.phase = 2;
    for (int i = 0; i < N_WORKERS; ++i) NN.worker_done[i] = 0;
    do {
        done = 1;
        for (int i = 0; i < N_WORKERS; ++i) 
            if (NN.worker_done[i] != 2) { 
                done = 0; 
                break; 
            }
        if (!done) _waitx(1);
    } while (!done);
    NN.phase = 0;
    
    // ----- PHASE 3: Conv3 -----
    NN.phase = 3;
    for (int i = 0; i < N_WORKERS; ++i) NN.worker_done[i] = 0;
    do {
        done = 1;
        for (int i = 0; i < N_WORKERS; ++i) 
            if (NN.worker_done[i] != 3) { 
                done = 0; 
                break; 
            }
        if (!done) _waitx(1);
    } while (!done);
    NN.phase = 0;
    
    // ----- PHASE 4: Conv4 -----
    NN.phase = 4;
    for (int i = 0; i < N_WORKERS; ++i) NN.worker_done[i] = 0;
    do {
        done = 1;
        for (int i = 0; i < N_WORKERS; ++i) 
            if (NN.worker_done[i] != 4) { 
                done = 0; 
                break; 
            }
        if (!done) _waitx(1);
    } while (!done);
    NN.phase = 0;
    
    // ----- PHASE 5: Global Average Pooling -----
    NN.phase = 5;
    for (int i = 0; i < N_WORKERS; ++i) NN.worker_done[i] = 0;
    do {
        done = 1;
        for (int i = 0; i < N_WORKERS; ++i) 
            if (NN.worker_done[i] != 5) { 
                done = 0; 
                break; 
            }
        if (!done) _waitx(1);
    } while (!done);
    NN.phase = 0;
    
    // ----- Dense Layer (Main cog) with INT8 weights -----
    for (int o = 0; o < DENSE_OUT; ++o) {
        float sum = dense_biases[o];
        for (int i = 0; i < GAP_OUT; ++i) {
            int weight_idx = i * DENSE_OUT + o;  // Row-major: [input][output]
            // Dequantize INT8 weight
            float weight = dequantize_int8(dense_weights[weight_idx], dense_scale);
            sum += NN.gap[i] * weight;
        }
        NN.out[o] = sum;
    }
    
    uint64_t ms_end = _getms();
    float ms_total = (float)(ms_end - ms_start);
    
    // Apply softmax and get prediction
    float logits[DENSE_OUT];
    for (int k = 0; k < DENSE_OUT; ++k) 
        logits[k] = NN.out[k];
    
    softmax_f(logits, DENSE_OUT);
    
    int argmax = 0;
    for (int i = 1; i < DENSE_OUT; ++i) 
        if (logits[i] > logits[argmax]) 
            argmax = i;
    
    // Display results
    printf("\nPredicted digit: %d\n", argmax);
    printf("Probabilities:\n");
    for (int i = 0; i < DENSE_OUT; ++i) {
        printf("  %d: %.6f", i, logits[i]);
        if (i == argmax) printf("  <-- PREDICTION");
        if (i == SAMPLE_LABEL) printf("  (TRUE LABEL)");
        printf("\n");
    }
    
    printf("\nCorrect: %s\n", (argmax == SAMPLE_LABEL) ? "YES" : "NO");
    printf("Inference time: %.3f ms\n", ms_total);
    printf("Memory saved: %.1f KB (%.1fx compression)\n", 
           ORIGINAL_SIZE_KB - QUANTIZED_SIZE_KB, COMPRESSION_RATIO);
    
    // Stop cogs
    _cogstop(cid_input);
    for (int i = 0; i < N_WORKERS; ++i) 
        _cogstop(cid_workers[i]);
    
    // Memory usage summary
    printf("\nMemory usage summary:\n");
    printf("  conv1_out: %d floats = %d bytes\n", C1_OUT*C1_H*C1_W, C1_OUT*C1_H*C1_W*4);
    printf("  conv2_out: %d floats = %d bytes\n", C2_OUT*C2_H*C2_W, C2_OUT*C2_H*C2_W*4);
    printf("  conv3_out: %d floats = %d bytes\n", C3_OUT*C3_H*C3_W, C3_OUT*C3_H*C3_W*4);
    printf("  conv4_out: %d floats = %d bytes\n", C4_OUT*C4_H*C4_W, C4_OUT*C4_H*C4_W*4);
    printf("  Total intermediate storage: %d bytes\n", 
           (C1_OUT*C1_H*C1_W + C2_OUT*C2_H*C2_W + C3_OUT*C3_H*C3_W + C4_OUT*C4_H*C4_W)*4);
    
    printf("\n=== INT8 Quantized Inference Complete ===\n");
    
    for (;;) _waitx(1000);
    return 0;
}
