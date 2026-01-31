#include <propeller2.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/vfs.h>
#include "numero_matriz.h"

struct __using("psram.spin2") psram;

#define IN_H 28
#define IN_W 28
#define IN_C 1

#define C1_OUT 32
#define C1_K 3
#define C1_H 26
#define C1_W 26
#define C1_P_H 13
#define C1_P_W 13

#define C2_OUT 64
#define C2_K 3
#define C2_H 11
#define C2_W 11
#define C2_P_H 5
#define C2_P_W 5

#define FC_IN (64 * 5 * 5)
#define FC1_OUT 128
#define FC2_OUT 10


#define VERY_NEGATIVE_FLOAT (-1e30f)

#define PSRAM_ADDR 0

#define OFF_conv1_w 0u
#define OFF_conv1_scale 288u
#define OFF_conv1_bias 292u

#define OFF_conv2_w 420u
#define OFF_conv2_scale 18852u
#define OFF_conv2_bias 18856u

#define OFF_fc1_w 19112u
#define OFF_fc1_scale 223912u
#define OFF_fc1_bias 223916u

#define OFF_fc2_w 224428u
#define OFF_fc2_scale 225708u
#define OFF_fc2_bias 225712u

#define TOTAL_WEIGHTS_BYTES 225752u

#define N_WORKERS 4
#define STACKSIZE 1024

unsigned char stack_input[STACKSIZE];
unsigned char stacks_workers[N_WORKERS][STACKSIZE];

typedef struct {
    volatile float img2d[28][28];
    volatile float c1p[32 * 13 * 13];
    volatile float c2p[64 * 5 * 5];
    volatile float fc1[128];
    volatile float out[10];
    volatile int flag_input;
    volatile int phase;
    volatile int worker_done[N_WORKERS];
} SharedNN;

SharedNN NN;

typedef struct {
    SharedNN *shared;
    int id;
} WorkerParam;

static inline float relu_inline(float x) { return x < 0.0f ? 0.0f : x; }

//Consider removing softmax, no need to show probabilities. Enough with searching the max value
static void softmax_f(float *x, int len) {
    float m = x[0];
    for (int i = 1; i < len; i++) if (x[i] > m) m = x[i];
    float s = 0.0f;
    for (int i = 0; i < len; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    for (int i = 0; i < len; i++) x[i] /= s;
}

static inline void psram_read_to(void *d, uint32_t a, size_t b) {
    psram.read((uint32_t)d, a, b);
}
static inline void psram_write_from(void *s, uint32_t a, size_t b) {
    psram.write((uint32_t)s, a, b);
}
static inline float psram_read_float(uint32_t a) {
    float t;
    psram_read_to(&t, a, 4);
    return t;
}

#define SD_READ_BUFFER 4096
static uint8_t sd_buf[SD_READ_BUFFER];

int load_weights_bin_to_psram(const char *filename, uint32_t base) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    long fsz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint32_t addr = base;
    size_t r;
    while ((r = fread(sd_buf, 1, SD_READ_BUFFER, fp)) > 0) {
        psram_write_from(sd_buf, addr, r);
        addr += r;
    }
    fclose(fp);
    return (int)fsz;
}

void cog_worker(void *p) {
    WorkerParam *wp = (WorkerParam *)p;
    SharedNN *s = wp->shared;
    int id = wp->id;

    int c1_per = (32 + N_WORKERS - 1) / N_WORKERS;
    int c1_start = id * c1_per;
    int c1_end = c1_start + c1_per;
    if (c1_end > 32) c1_end = 32;

    int c2_per = (64 + N_WORKERS - 1) / N_WORKERS;
    int c2_start = id * c2_per;
    int c2_end = c2_start + c2_per;
    if (c2_end > 64) c2_end = 64;

    int fc1_per = (128 + N_WORKERS - 1) / N_WORKERS;
    int fc1_start = id * fc1_per;
    int fc1_end = fc1_start + fc1_per;
    if (fc1_end > 128) fc1_end = 128;

    int8_t wq_conv1[9];
    float wf_conv1[9];

    int8_t wq_conv2[9];
    float wf_conv2[9];

    int8_t fc1_chunk_q[512];

    while (1) {
        int phase = s->phase;

        if (phase == 1) {
            float sc = psram_read_float(PSRAM_ADDR + OFF_conv1_scale);
            for (int oc = c1_start; oc < c1_end; oc++) {
                uint32_t off = PSRAM_ADDR + OFF_conv1_w + oc * 9;
                psram_read_to(wq_conv1, off, 9);
                for (int i = 0; i < 9; i++) wf_conv1[i] = (float)wq_conv1[i] * sc;
                float b;
                psram_read_to(&b, PSRAM_ADDR + OFF_conv1_bias + oc * 4, 4);
                for (int oy = 0; oy < 13; oy++) {
                    for (int ox = 0; ox < 13; ox++) {
                        float mv = VERY_NEGATIVE_FLOAT;
                        for (int py = 0; py < 2; py++) {
                            for (int px = 0; px < 2; px++) {
                                int cy = oy * 2 + py;
                                int cx = ox * 2 + px;
                                float v =
                                    b +
                                    s->img2d[cy + 0][cx + 0] * wf_conv1[0] +
                                    s->img2d[cy + 0][cx + 1] * wf_conv1[1] +
                                    s->img2d[cy + 0][cx + 2] * wf_conv1[2] +
                                    s->img2d[cy + 1][cx + 0] * wf_conv1[3] +
                                    s->img2d[cy + 1][cx + 1] * wf_conv1[4] +
                                    s->img2d[cy + 1][cx + 2] * wf_conv1[5] +
                                    s->img2d[cy + 2][cx + 0] * wf_conv1[6] +
                                    s->img2d[cy + 2][cx + 1] * wf_conv1[7] +
                                    s->img2d[cy + 2][cx + 2] * wf_conv1[8];
                                v = relu_inline(v);
                                if (v > mv) mv = v;
                            }
                        }
                        s->c1p[(oc * 13 + oy) * 13 + ox] = mv;
                    }
                }
            }
            s->worker_done[id] = 1;
            while (s->phase == 1) _waitx(1);
        }

        if (phase == 2) {
            float sc = psram_read_float(PSRAM_ADDR + OFF_conv2_scale);
            for (int oc = c2_start; oc < c2_end; oc++) {
                float b;
                psram_read_to(&b, PSRAM_ADDR + OFF_conv2_bias + oc * 4, 4);
                for (int oy = 0; oy < 5; oy++) {
                    for (int ox = 0; ox < 5; ox++) {
                        float mv = VERY_NEGATIVE_FLOAT;
                        for (int py = 0; py < 2; py++) {
                            for (int px = 0; px < 2; px++) {
                                int cy = oy * 2 + py;
                                int cx = ox * 2 + px;
                                float sum = b;
                                for (int ic = 0; ic < 32; ic++) {
                                    uint32_t off = PSRAM_ADDR + OFF_conv2_w + (oc * 32 + ic) * 9;
                                    psram_read_to(wq_conv2, off, 9);
                                    for (int i = 0; i < 9; i++) wf_conv2[i] = (float)wq_conv2[i] * sc;
                                    const float *f = &s->c1p[(ic * 13 + cy) * 13 + cx];
                                    sum +=
                                        f[0] * wf_conv2[0] +
                                        f[1] * wf_conv2[1] +
                                        f[2] * wf_conv2[2] +
                                        f[13 + 0] * wf_conv2[3] +
                                        f[13 + 1] * wf_conv2[4] +
                                        f[13 + 2] * wf_conv2[5] +
                                        f[26 + 0] * wf_conv2[6] +
                                        f[26 + 1] * wf_conv2[7] +
                                        f[26 + 2] * wf_conv2[8];
                                }
                                float v = relu_inline(sum);
                                if (v > mv) mv = v;
                            }
                        }
                        s->c2p[(oc * 5 + oy) * 5 + ox] = mv;
                    }
                }
            }
            s->worker_done[id] = 2;
            while (s->phase == 2) _waitx(1);
        }

        if (phase == 3) {
            float sc = psram_read_float(PSRAM_ADDR + OFF_fc1_scale);
            for (int o = fc1_start; o < fc1_end; o++) {
                float b;
                psram_read_to(&b, PSRAM_ADDR + OFF_fc1_bias + o * 4, 4);
                float sum = b;
                uint32_t base = PSRAM_ADDR + OFF_fc1_w + o * FC_IN;
                int remaining = FC_IN;
                int idx = 0;
                while (remaining > 0) {
                    int r = remaining > 512 ? 512 : remaining;
                    psram_read_to(fc1_chunk_q, base + idx, r);
                    for (int k = 0; k < r; k++) {
                        float w = (float)fc1_chunk_q[k] * sc;
                        sum += s->c2p[idx + k] * w;
                    }
                    idx += r;
                    remaining -= r;
                }
                s->fc1[o] = relu_inline(sum);
            }
            s->worker_done[id] = 3;
            while (s->phase == 3) _waitx(1);
        }

        _waitx(10);
    }
}

void cog_input(void *p) {
    SharedNN *s = (SharedNN *)p;
    while (1) {
        if (s->flag_input == 2) {
            for (int y = 0; y < 28; y++)
                for (int x = 0; x < 28; x++)
                    s->img2d[y][x] = numero[y][x] / 255.0f;
            s->flag_input = 1;
        }
        _waitx(100);
    }
}

int main(void) {
    _setbaud(230400);
    printf("Starting CNN inference...\n");

    int r = psram.start();
    if (r < 0) {
        printf("PSRAM initialization error: %d\n", r);
        while(1);
    }
    printf("PSRAM initialized\n");

    int m = _mount("/sd", _vfs_open_sdcard());
    if (m < 0) {
        printf("SD card mount failed: %d\n", m);
        while(1);
    }
    printf("SD card mounted\n");

    int loaded = load_weights_bin_to_psram("/sd/weights.bin", PSRAM_ADDR);
    if (loaded <= 0) {
        printf("Failed to load weights: %d\n", loaded);
        while(1);
    }
    printf("Weights loaded: %d bytes\n", loaded);
    
    printf("Verifying first weight...\n");
    int8_t test_weight[10];
    float test_scale;
    psram_read_to(test_weight, PSRAM_ADDR + OFF_conv1_w, 10);
    psram_read_to(&test_scale, PSRAM_ADDR + OFF_conv1_scale, 4);
    printf("First conv1 weights (int8): ");
    for (int i = 0; i < 10; i++) printf("%d ", test_weight[i]);
    printf("\nconv1 scale: %.10f\n", test_scale);
    printf("First dequantized weight: %.10f\n", (float)test_weight[0] * test_scale);

    memset(&NN, 0, sizeof(NN));
    NN.flag_input = 0;
    NN.phase = 0;

    int cid_in = _cogstart_C(cog_input, &NN, stack_input, sizeof(stack_input));

    WorkerParam params[N_WORKERS];
    int cids[N_WORKERS];
    for (int i = 0; i < N_WORKERS; i++) {
        params[i].shared = &NN;
        params[i].id = i;
        cids[i] = _cogstart_C(cog_worker, &params[i], stacks_workers[i], sizeof(stacks_workers[i]));
    }

    printf("Loading image data...\n");
    NN.flag_input = 2;
    while (NN.flag_input != 1) _waitx(1);

    uint64_t ms0 = _getms();

    printf("Phase 1: Conv1 + Pool1...\n");
    for (int i = 0; i < N_WORKERS; i++) NN.worker_done[i] = 0;
    NN.phase = 1;
    for (;;) {
        int ok = 1;
        for (int i = 0; i < N_WORKERS; i++) if (NN.worker_done[i] != 1) ok = 0;
        if (ok) break;
        _waitx(1);
    }
    NN.phase = 0;

    printf("Phase 2: Conv2 + Pool2...\n");
    for (int i = 0; i < N_WORKERS; i++) NN.worker_done[i] = 0;
    NN.phase = 2;
    for (;;) {
        int ok = 1;
        for (int i = 0; i < N_WORKERS; i++) if (NN.worker_done[i] != 2) ok = 0;
        if (ok) break;
        _waitx(1);
    }
    NN.phase = 0;

    printf("Phase 3: FC1...\n");
    for (int i = 0; i < N_WORKERS; i++) NN.worker_done[i] = 0;
    NN.phase = 3;
    for (;;) {
        int ok = 1;
        for (int i = 0; i < N_WORKERS; i++) if (NN.worker_done[i] != 3) ok = 0;
        if (ok) break;
        _waitx(1);
    }
    NN.phase = 0;

    printf("FC2 layer...\n");
    float sc2 = psram_read_float(PSRAM_ADDR + OFF_fc2_scale);
    int8_t row128[128];
    for (int o = 0; o < 10; o++) {
        float b;
        psram_read_to(&b, PSRAM_ADDR + OFF_fc2_bias + o * 4, 4);
        float sum = b;
        uint32_t off = PSRAM_ADDR + OFF_fc2_w + o * 128;
        psram_read_to(row128, off, 128);
        for (int i = 0; i < 128; i++) {
            float w = (float)row128[i] * sc2;
            sum += NN.fc1[i] * w;
        }
        NN.out[o] = sum;
    }

    uint64_t ms1 = _getms();

    printf("Softmax...\n");
    float logits[10];
    for (int i = 0; i < 10; i++) logits[i] = NN.out[i];
    
    printf("Raw outputs before softmax:\n");
    for (int i = 0; i < 10; i++) {
        printf("  %d: %.6f\n", i, NN.out[i]);
    }
    
    softmax_f(logits, 10);
    int argmax = 0;
    for (int i = 1; i < 10; i++) if (logits[i] > logits[argmax]) argmax = i;

    printf("\n=== RESULTS ===\n");
    printf("Predicted digit: %d\n", argmax);
    printf("Probabilities:\n");
    for (int i = 0; i < 10; i++) {
        printf("  %d: %.4f", i, logits[i]);
        if (i == argmax) printf("  <--");
        printf("\n");
    }
    printf("Inference time: %d ms\n", (int)(ms1 - ms0));

    _cogstop(cid_in);
    for (int i = 0; i < N_WORKERS; i++) _cogstop(cids[i]);

    printf("Done.\n");
    while (1) _waitx(1000);
    return 0;
}
