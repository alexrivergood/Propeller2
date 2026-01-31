#include <stdio.h>
#include <propeller2.h>
#include <math.h>
#include <iris_weights.h>

#define STACKSIZE 256

unsigned char stack1[STACKSIZE];
unsigned char stack2[STACKSIZE];
unsigned char stack3[STACKSIZE];

typedef struct {
    volatile float input[4];
    volatile float h1[16];
    volatile float h2[12];
    volatile float output[3];
    volatile int layer1_done;
    volatile int layer2_done;
    volatile int output_done;
} SharedNN;

SharedNN nn;

static inline float relu(float x) { return x < 0 ? 0 : x; }

static const float mean[4] = {5.843333f, 3.057333f, 3.758000f, 1.199333f};
static const float stdv[4] = {0.825301f, 0.434411f, 1.759404f, 0.759693f};

void cog_layer1(void *p) {
    SharedNN *s = (SharedNN *)p;
    
    for (int j = 0; j < 16; j++) {
        float sum = B0[j];
        for (int i = 0; i < 4; i++)
            sum += s->input[i] * W0[i][j];
        s->h1[j] = relu(sum);
    }
    
    s->layer1_done = 1;
    _cogstop(_cogid());
}

void cog_layer2(void *p) {
    SharedNN *s = (SharedNN *)p;
    
    while (!s->layer1_done) {
        _waitx(_clockfreq() / 1000);
    }
    
    for (int j = 0; j < 12; j++) {
        float sum = B1[j];
        for (int i = 0; i < 16; i++)
            sum += s->h1[i] * W1[i][j];
        s->h2[j] = relu(sum);
    }
    
    s->layer2_done = 1;
    _cogstop(_cogid());
}

void cog_output(void *p) {
    SharedNN *s = (SharedNN *)p;
    
    while (!s->layer2_done) {
        _waitx(_clockfreq() / 1000);
    }
    
    for (int k = 0; k < 3; k++) {
        float sum = B2[k];
        for (int j = 0; j < 12; j++)
            sum += s->h2[j] * W2[j][k];
        s->output[k] = sum;
    }
    
    s->output_done = 1;
    _cogstop(_cogid());
}

void softmax(float *x, int len) {
    float max = x[0];
    for (int i = 1; i < len; i++) if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < len; i++) x[i] /= sum;
}

void main() {
    const char *names[] = {"Setosa", "Versicolor", "Virginica"};
    
    float test_inputs[15][4] = {
        {5.1f, 3.5f, 1.4f, 0.2f},
        {4.9f, 3.0f, 1.4f, 0.2f},
        {5.0f, 3.4f, 1.5f, 0.2f},
        {4.6f, 3.1f, 1.5f, 0.2f},
        {5.4f, 3.9f, 1.7f, 0.4f},
        {6.0f, 2.2f, 4.0f, 1.0f},
        {5.9f, 3.0f, 4.2f, 1.5f},
        {6.1f, 2.8f, 4.0f, 1.3f},
        {6.2f, 2.9f, 4.3f, 1.3f},
        {5.6f, 2.9f, 3.6f, 1.3f},
        {6.5f, 3.0f, 5.8f, 2.2f},
        {7.1f, 3.0f, 5.9f, 2.1f},
        {6.9f, 3.1f, 5.4f, 2.1f},
        {6.7f, 3.3f, 5.7f, 2.5f},
        {7.2f, 3.6f, 6.1f, 2.5f}
    };
    
    for (int t = 0; t < 15; t++) {
        nn.layer1_done = nn.layer2_done = nn.output_done = 0;
        
        for (int i = 0; i < 4; i++) {
            nn.input[i] = (test_inputs[t][i] - mean[i]) / stdv[i];
        }
        
        uint64_t t_start = _getcnt();
        
        int cid1 = _cogstart_C(cog_layer1, &nn, stack1, sizeof(stack1));
        int cid2 = _cogstart_C(cog_layer2, &nn, stack2, sizeof(stack2));
        int cid3 = _cogstart_C(cog_output, &nn, stack3, sizeof(stack3));
        
        while (!nn.output_done) {
            _waitx(_clockfreq() / 1000);
        }
        
        uint64_t t_end = _getcnt();
        uint64_t ticks = t_end - t_start;
        float ms = (float)ticks * 1000.0f / (float)_clockfreq();
        
        float out[3] = { nn.output[0], nn.output[1], nn.output[2] };
        softmax(out, 3);
        
        int max_i = 0;
        for (int i = 1; i < 3; i++) if (out[i] > out[max_i]) max_i = i;
        
        printf("\nTest sample %2d:\n", t + 1);
        printf("  Setosa:     %.4f\n", out[0]);
        printf("  Versicolor: %.4f\n", out[1]);
        printf("  Virginica:  %.4f\n", out[2]);
        printf("  Predicted: %s\n", names[max_i]);
        printf("  Inference time: %.3f ms\n", ms);
    }
    
    printf("\nAll inferences completed\n");
    for (;;) _waitx(_clockfreq());
}
