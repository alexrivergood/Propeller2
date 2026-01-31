//This is an unoptimized structure of a CNN in C, where the weights have been exported
//The weight exceeds the 512KB of the HUB Ram by double
//No use of parallelism

#include <stdio.h>
#include <propeller2.h>
#include <math.h>
#include <mnist_cnn_weights.h>   
#include <numero_vector.h>       

#define STACKSIZE 512


unsigned char stack_input[STACKSIZE];
unsigned char stack_conv[STACKSIZE];
unsigned char stack_output[STACKSIZE];



#define IN_H 28
#define IN_W 28
#define IN_C 1


#define C1_OUT 32
#define C1_K 3
#define C1_H (IN_H - (C1_K - 1))     // 28 - 2 = 26
#define C1_W (IN_W - (C1_K - 1))     // 26
#define C1_P_H (C1_H / 2)            // after 2x2 max pool -> 13
#define C1_P_W (C1_W / 2)


#define C2_OUT 64
#define C2_K 3
#define C2_H (C1_P_H - (C2_K - 1))   // 13 - 2 = 11
#define C2_W (C1_P_W - (C2_K - 1))   // 11
#define C2_P_H (C2_H / 2)            // -> 5 (integer)
#define C2_P_W (C2_W / 2)            // -> 5


#define FC_IN (C2_OUT * C2_P_H * C2_P_W)   // 64 * 5 * 5 = 1600
#define FC1_OUT 128
#define FC2_OUT 10


typedef struct {
    volatile float input[IN_H * IN_W];           
    volatile float c1p[C1_OUT * C1_P_H * C1_P_W]; 
    volatile float c2p[C2_OUT * C2_P_H * C2_P_W]; 
    volatile float fc1[FC1_OUT];
    volatile float output[FC2_OUT];
    volatile int ready_in;    
    volatile int ready_c1;    
    volatile int ready_c2;    
    volatile int ready_out;   
} SharedNN;

SharedNN nn;


static inline float relu_f(float x) { return x < 0.0f ? 0.0f : x; }

void softmax_f(float *x, int len) {
    float maxv = x[0];
    for (int i = 1; i < len; ++i) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        x[i] = expf(x[i] - maxv);
        sum += x[i];
    }
    for (int i = 0; i < len; ++i) x[i] /= sum;
}





static inline int idx_conv1_w(int out, int in, int kh, int kw) {
    return ((out * IN_C + in) * C1_K + kh) * C1_K + kw;
}

static inline int idx_conv2_w(int out, int in, int kh, int kw) {
    return ((out * C1_OUT + in) * C2_K + kh) * C2_K + kw;
}

static inline int idx_fc1_w(int out, int in) {
    return out * FC_IN + in;
}

static inline int idx_fc2_w(int out, int in) {
    return out * FC1_OUT + in;
}

void conv2d_valid_multichannel(const float *in, int in_c, int in_h, int in_w,
                               float *out, int out_c, int k, const float *weights, const float *bias,
                               int (*windex)(int,int,int,int)) 
    int out_h = in_h - (k - 1);
    int out_w = in_w - (k - 1);
    for (int oc = 0; oc < out_c; ++oc) {
        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox = 0; ox < out_w; ++ox) {
                float s = bias ? bias[oc] : 0.0f;
                for (int ic = 0; ic < in_c; ++ic) {
                    for (int ky = 0; ky < k; ++ky) {
                        for (int kx = 0; kx < k; ++kx) {
                            int in_y = oy + ky;
                            int in_x = ox + kx;
                            int in_idx = (ic * in_h + in_y) * in_w + in_x;
                            int widx = windex(oc, ic, ky, kx);
                            s += in[in_idx] * weights[widx];
                        }
                    }
                }
                int out_idx = (oc * out_h + oy) * out_w + ox;
                out[out_idx] = s;
            }
        }
    }
}
void maxpool2x2(const float *in, int c, int in_h, int in_w, float *out) {
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    for (int ch = 0; ch < c; ++ch) {
        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox = 0; ox < out_w; ++ox) {
                int y0 = oy * 2;
                int x0 = ox * 2;
                int idx00 = (ch * in_h + y0) * in_w + x0;
                int idx01 = idx00 + 1;
                int idx10 = idx00 + in_w;
                int idx11 = idx10 + 1;
                float a = in[idx00];
                float b = in[idx01];
                float c0 = in[idx10];
                float d = in[idx11];
                float m = a;
                if (b > m) m = b;
                if (c0 > m) m = c0;
                if (d > m) m = d;
                out[(ch * out_h + oy) * out_w + ox] = m;
            }
        }
    }
}

void cog_input(void *p) {
    SharedNN *s = (SharedNN *)p;
    while (1) {
        if (s->ready_in == 2) {
            for (int i = 0; i < IN_H * IN_W; ++i) s->input[i] = numero[i];
            s->ready_in = 1;
        }
        _waitx(_clockfreq() / 1000);
    }
}
void cog_conv_pool_fc1(void *p) {
    SharedNN *s = (SharedNN *)p;

    static float conv1_buf[C1_OUT * C1_H * C1_W]; 
    

    while (1) {
        if (s->ready_in == 1) {
            conv2d_valid_multichannel((const float *)s->input, IN_C, IN_H, IN_W,
                                      conv1_buf, C1_OUT, C1_K,
                                      conv1_weight, conv1_bias, idx_conv1_w);

            for (int oc = 0; oc < C1_OUT; ++oc) {
                for (int oy = 0; oy < C1_P_H; ++oy) {
                    for (int ox = 0; ox < C1_P_W; ++ox) {
                        int y0 = oy * 2;
                        int x0 = ox * 2;
                        int idx00 = (oc * C1_H + y0) * C1_W + x0;
                        int idx01 = idx00 + 1;
                        int idx10 = idx00 + C1_W;
                        int idx11 = idx10 + 1;
                        float a = relu_f(conv1_buf[idx00]);
                        float b = relu_f(conv1_buf[idx01]);
                        float c0 = relu_f(conv1_buf[idx10]);
                        float d = relu_f(conv1_buf[idx11]);
                        float m = a;
                        if (b > m) m = b;
                        if (c0 > m) m = c0;
                        if (d > m) m = d;
                        s->c1p[(oc * C1_P_H + oy) * C1_P_W + ox] = m;
                    }
                }
            }

            static float conv2_buf[C2_OUT * C2_H * C2_W]; 
            conv2d_valid_multichannel((const float *)s->c1p, C1_OUT, C1_P_H, C1_P_W,
                                      conv2_buf, C2_OUT, C2_K,
                                      conv2_weight, conv2_bias, idx_conv2_w);

            for (int oc = 0; oc < C2_OUT; ++oc) {
                for (int oy = 0; oy < C2_P_H; ++oy) {
                    for (int ox = 0; ox < C2_P_W; ++ox) {
                        int y0 = oy * 2;
                        int x0 = ox * 2;
                        int idx00 = (oc * C2_H + y0) * C2_W + x0;
                        int idx01 = idx00 + 1;
                        int idx10 = idx00 + C2_W;
                        int idx11 = idx10 + 1;
                        float a = relu_f(conv2_buf[idx00]);
                        float b = relu_f(conv2_buf[idx01]);
                        float c0 = relu_f(conv2_buf[idx10]);
                        float d = relu_f(conv2_buf[idx11]);
                        float m = a;
                        if (b > m) m = b;
                        if (c0 > m) m = c0;
                        if (d > m) m = d;
                        s->c2p[(oc * C2_P_H + oy) * C2_P_W + ox] = m;
                    }
                }
            }

            
            for (int o = 0; o < FC1_OUT; ++o) {
                float sum = fc1_bias[o];
                
                for (int i = 0; i < FC_IN; ++i) {
                    sum += s->c2p[i] * fc1_weight[idx_fc1_w(o, i)];
                }
                s->fc1[o] = relu_f(sum);
            }

            
            s->ready_c1 = 1;
            s->ready_in = 0;
        }
        _waitx(_clockfreq() / 1000);
    }
}


void cog_fc2_output(void *p) {
    SharedNN *s = (SharedNN *)p;
    while (1) {
        if (s->ready_c1 == 1) {
            
            for (int o = 0; o < FC2_OUT; ++o) {
                float sum = fc2_bias[o];
                for (int i = 0; i < FC1_OUT; ++i) {
                    sum += s->fc1[i] * fc2_weight[idx_fc2_w(o, i)];
                }
                s->output[o] = sum;
            }
            s->ready_out = 1;
            s->ready_c1 = 0;
        }
        _waitx(_clockfreq() / 1000);
    }
}

void main() {
    printf("=== MNIST CNN Inference on Propeller 2 (FlexProp) ===\n");
    nn.ready_in = nn.ready_c1 = nn.ready_c2 = nn.ready_out = 0;
    int cid_in = _cogstart_C(cog_input, &nn, stack_input, sizeof(stack_input));
    int cid_conv = _cogstart_C(cog_conv_pool_fc1, &nn, stack_conv, sizeof(stack_conv));
    int cid_out = _cogstart_C(cog_fc2_output, &nn, stack_output, sizeof(stack_output));

    printf("COGs started: input=%d conv=%d output=%d\n", cid_in, cid_conv, cid_out);
    for (int i = 0; i < IN_H * IN_W; ++i) {
        nn.input[i] = numero[i];
    }

    
    nn.ready_in = 2;   
    nn.ready_out = 0;

    uint64_t t_start = _getcnt();

    
    while (!nn.ready_out) _waitx(_clockfreq() / 1000);

    uint64_t t_end = _getcnt();
    float ms = (float)(t_end - t_start) * 1000.0f / (float)_clockfreq();

    float out[FC2_OUT];
    for (int k = 0; k < FC2_OUT; ++k) out[k] = nn.output[k];
    softmax_f(out, FC2_OUT);

    int argmax = 0;
    for (int i = 1; i < FC2_OUT; ++i) if (out[i] > out[argmax]) argmax = i;

    printf("\nPredicted digit: %d\n", argmax);
    printf("Probabilities:\n");
    for (int i = 0; i < FC2_OUT; ++i) printf("  %d: %.4f\n", i, out[i]);
    printf("Inference time: %.3f ms\n", ms);

    _cogstop(cid_in);
    _cogstop(cid_conv);
    _cogstop(cid_out);

    printf("\n=== Inference finished ===\n");
    for (;;) _waitx(_clockfreq());
}
