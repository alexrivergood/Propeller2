#include <stdio.h>
#include <propeller2.h>
#include <math.h>
#include <iris_weights.h>

#define STACKSIZE 256

typedef struct {
    float input[4];
    float h1[16];
    float h2[12];
    float output[3];
} NetworkData;

static inline float relu(float x) { return x < 0 ? 0 : x; }

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

static const float mean[4] = {5.843333f, 3.057333f, 3.758000f, 1.199333f};
static const float stdv[4] = {0.825301f, 0.434411f, 1.759404f, 0.759693f};

void run_neural_network(float test_input[4], float output[3]) {
    NetworkData data;
    
    for (int i = 0; i < 4; i++) {
        data.input[i] = (test_input[i] - mean[i]) / stdv[i];
    }
    
    for (int j = 0; j < 16; j++) {
        float sum = B0[j];
        for (int i = 0; i < 4; i++)
            sum += data.input[i] * W0[i][j];
        data.h1[j] = relu(sum);
    }
    
    for (int j = 0; j < 12; j++) {
        float sum = B1[j];
        for (int i = 0; i < 16; i++)
            sum += data.h1[i] * W1[i][j];
        data.h2[j] = relu(sum);
    }
    
    for (int k = 0; k < 3; k++) {
        float sum = B2[k];
        for (int j = 0; j < 12; j++)
            sum += data.h2[j] * W2[j][k];
        data.output[k] = sum;
    }
    
    output[0] = data.output[0];
    output[1] = data.output[1];
    output[2] = data.output[2];
}

void main() {
    printf("Single COG Neural Network Inference\n");
    
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
        uint64_t t_start = _getcnt();
        
        float output[3];
        run_neural_network(test_inputs[t], output);
        
        uint64_t t_end = _getcnt();
        uint64_t ticks = t_end - t_start;
        float ms = (float)ticks * 1000.0f / (float)_clockfreq();
        
        softmax(output, 3);
        
        int max_i = 0;
        for (int i = 1; i < 3; i++) if (output[i] > output[max_i]) max_i = i;
        
        printf("\nTest sample %2d:\n", t + 1);
        printf("  Setosa:     %.4f\n", output[0]);
        printf("  Versicolor: %.4f\n", output[1]);
        printf("  Virginica:  %.4f\n", output[2]);
        printf("  Predicted: %s\n", names[max_i]);
        printf("  Inference time: %.3f ms\n", ms);
    }
    
    printf("\nAll inferences completed\n");
    for (;;) _waitx(_clockfreq());
}
