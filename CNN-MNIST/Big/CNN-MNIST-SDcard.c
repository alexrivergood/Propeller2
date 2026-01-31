//Inference time 6s
//Loads weights from the sdcard to the PSRAM
//Small samples are taken from the PSRAM for the calculations, then replaced for the next calculations

#include <stdio.h>
#include <propeller2.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <sys/vfs.h>
#include <stdlib.h>
#include "numero_matriz.h"

struct __using("psram.spin2") psram;

#define BUFFER_SIZE 4096
#define PSRAM_SIZE_BYTES (32u * 1024u * 1024u)

#define N_WORKERS 4
#define STACKSIZE 1024
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

#define CONV1_W_OFFSET 0
#define CONV1_W_SIZE 288
#define CONV1_SCALE_OFFSET 288
#define CONV1_BIAS_OFFSET 292

#define CONV2_W_OFFSET 420
#define CONV2_W_SIZE 18432
#define CONV2_SCALE_OFFSET 18852
#define CONV2_BIAS_OFFSET 18856

#define FC1_W_OFFSET 19112
#define FC1_W_SIZE 204800
#define FC1_SCALE_OFFSET 223912
#define FC1_BIAS_OFFSET 223916

#define FC2_W_OFFSET 224428
#define FC2_W_SIZE 1280
#define FC2_SCALE_OFFSET 225708
#define FC2_BIAS_OFFSET 225712

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


static float conv1_bf[C1_OUT];
static float conv2_bf[C2_OUT];
static float fc1_bf[FC1_OUT];
static float fc2_bf[FC2_OUT];

static float scale_conv1;
static float scale_conv2;
static float scale_fc1;
static float scale_fc2;

static uint8_t transfer_buffer[BUFFER_SIZE];
static int8_t load_buffer[BUFFER_SIZE];

static inline float relu_inline(float x) { return x < 0.0f ? 0.0f : x; }

static void softmax_f(float *x, int len) {
    float maxv = x[0];
    for (int i = 1; i < len; ++i) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) { x[i] = expf(x[i] - maxv); sum += x[i]; }
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

//De la SD a PSRAM funcion. Se utilizan un buffersize de 4096
int32_t load_file_to_psram(const char *filename, uint32_t psram_addr) {
    FILE *fp;
    size_t bytes_read;
    uint32_t total_bytes = 0;
    uint32_t current_addr = psram_addr;

    fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: Cannot open %s\n", filename);
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    printf("File size: %ld bytes\n", file_size);

    printf("Transferring to PSRAM\n");
    while ((bytes_read = fread(transfer_buffer, 1, BUFFER_SIZE, fp)) > 0)
    {
        psram.write((uint32_t)transfer_buffer, current_addr, bytes_read);
        current_addr += bytes_read;
        total_bytes += bytes_read;

        if (total_bytes % 102400 == 0) {
            printf(".");
        }
    }
    printf("\n");

    fclose(fp);
    printf("Transfer complete: %lu bytes\n", total_bytes);
    return total_bytes;
}

//Load only scales and biases (small) into HUB RAM; weights remain in PSRAM.
void load_weights_from_psram(void) {
    printf("Loading layer scales and biases (weights remain in PSRAM)\n");

    
    psram.read((uint32_t)&scale_conv1, CONV1_SCALE_OFFSET, sizeof(float));
    psram.read((uint32_t)conv1_bf, CONV1_BIAS_OFFSET, C1_OUT * sizeof(float));

    
    psram.read((uint32_t)&scale_conv2, CONV2_SCALE_OFFSET, sizeof(float));
    psram.read((uint32_t)conv2_bf, CONV2_BIAS_OFFSET, C2_OUT * sizeof(float));

    
    psram.read((uint32_t)&scale_fc1, FC1_SCALE_OFFSET, sizeof(float));
    psram.read((uint32_t)fc1_bf, FC1_BIAS_OFFSET, FC1_OUT * sizeof(float));

    
    psram.read((uint32_t)&scale_fc2, FC2_SCALE_OFFSET, sizeof(float));
    psram.read((uint32_t)fc2_bf, FC2_BIAS_OFFSET, FC2_OUT * sizeof(float));

    printf("Scales and biases loaded.\n");
}


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

    int fc2_per = (FC2_OUT + N_WORKERS - 1) / N_WORKERS;
    int fc2_start = id * fc2_per;
    int fc2_end = fc2_start + fc2_per; if (fc2_end > FC2_OUT) fc2_end = FC2_OUT;

    
    int8_t kbuf9[9]; 
    #define FC1_CHUNK 200
    int8_t fc1_chunk_buf[FC1_CHUNK];
    int8_t fc2_wbuf[FC1_OUT]; 

    while (1) {
        int phase = s->phase;

        if (phase == 1) {
            
            for (int oc = c1_start; oc < c1_end; ++oc) {
                float bias = conv1_bf[oc];
                int base_idx = idx_conv1_w(oc, 0, 0, 0); 
                psram.read((uint32_t)kbuf9, CONV1_W_OFFSET + base_idx, 9);
                for (int oy = 0; oy < C1_P_H; ++oy) {
                    for (int ox = 0; ox < C1_P_W; ++ox) {
                        float maxv = VERY_NEGATIVE_FLOAT;
                        for (int py = 0; py < 2; ++py) {
                            for (int px = 0; px < 2; ++px) {
                                int conv_y = oy * 2 + py;
                                int conv_x = ox * 2 + px;
                                float ssum = bias;
                                ssum += s->img2d[conv_y + 0][conv_x + 0] * ((float)kbuf9[0] * scale_conv1);
                                ssum += s->img2d[conv_y + 0][conv_x + 1] * ((float)kbuf9[1] * scale_conv1);
                                ssum += s->img2d[conv_y + 0][conv_x + 2] * ((float)kbuf9[2] * scale_conv1);
                                ssum += s->img2d[conv_y + 1][conv_x + 0] * ((float)kbuf9[3] * scale_conv1);
                                ssum += s->img2d[conv_y + 1][conv_x + 1] * ((float)kbuf9[4] * scale_conv1);
                                ssum += s->img2d[conv_y + 1][conv_x + 2] * ((float)kbuf9[5] * scale_conv1);
                                ssum += s->img2d[conv_y + 2][conv_x + 0] * ((float)kbuf9[6] * scale_conv1);
                                ssum += s->img2d[conv_y + 2][conv_x + 1] * ((float)kbuf9[7] * scale_conv1);
                                ssum += s->img2d[conv_y + 2][conv_x + 2] * ((float)kbuf9[8] * scale_conv1);

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
            int weights_per_oc = C1_OUT * 9;
            int8_t wbuf_per_oc[C1_OUT * 9];
            for (int oc = c2_start; oc < c2_end; ++oc) {
                int base_idx = idx_conv2_w(oc, 0, 0, 0); 
                psram.read((uint32_t)wbuf_per_oc, CONV2_W_OFFSET + base_idx, weights_per_oc);

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
                                    int wbase = ic * 9;
                                    const float *fmap = &s->c1p[(ic * C1_P_H + conv_y) * C1_P_W + conv_x];
                                    ssum += fmap[0] * ((float)wbuf_per_oc[wbase + 0] * scale_conv2)
                                         + fmap[1] * ((float)wbuf_per_oc[wbase + 1] * scale_conv2)
                                         + fmap[2] * ((float)wbuf_per_oc[wbase + 2] * scale_conv2)
                                         + fmap[C1_P_W+0] * ((float)wbuf_per_oc[wbase + 3] * scale_conv2)
                                         + fmap[C1_P_W+1] * ((float)wbuf_per_oc[wbase + 4] * scale_conv2)
                                         + fmap[C1_P_W+2] * ((float)wbuf_per_oc[wbase + 5] * scale_conv2)
                                         + fmap[2*C1_P_W+0] * ((float)wbuf_per_oc[wbase + 6] * scale_conv2)
                                         + fmap[2*C1_P_W+1] * ((float)wbuf_per_oc[wbase + 7] * scale_conv2)
                                         + fmap[2*C1_P_W+2] * ((float)wbuf_per_oc[wbase + 8] * scale_conv2);
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
                int in_idx = 0;
                while (in_idx < FC_IN) {
                    int chunk = (FC_IN - in_idx) > FC1_CHUNK ? FC1_CHUNK : (FC_IN - in_idx);
                    int base_idx = idx_fc1_w(o, in_idx);
                    psram.read((uint32_t)fc1_chunk_buf, FC1_W_OFFSET + base_idx, chunk);
                    for (int j = 0; j < chunk; ++j) {
                        sum += s->c2p[in_idx + j] * ((float)fc1_chunk_buf[j] * scale_fc1);
                    }
                    in_idx += chunk;
                }
                s->fc1[o] = relu_inline(sum);
            }
            s->worker_done[id] = 3;
            while (s->phase == 3) _waitx(1);
            continue;
        }

        if (phase == 4) {
            for (int o = fc2_start; o < fc2_end; ++o) {
                float sum = fc2_bf[o];
                int base_idx = idx_fc2_w(o, 0); /* index in int8 array */
                /* read 128 bytes */
                psram.read((uint32_t)fc2_wbuf, FC2_W_OFFSET + base_idx, FC1_OUT);
                for (int i = 0; i < FC1_OUT; ++i) {
                    sum += s->fc1[i] * ((float)fc2_wbuf[i] * scale_fc2);
                }
                s->out[o] = sum;
            }
            s->worker_done[id] = 4;
            while (s->phase == 4) _waitx(1);
            continue;
        }
        _waitx(100);
    }
}

int main() {
    _setbaud(230400);

    printf("CNN with PSRAM Weights (on-demand PSRAM reads)\n");

    int r = psram.start();
    if (r < 0) {
        printf("PSRAM initialization error: %d\n", r);
        while(1);
    }
    printf("PSRAM initialized\n");

    int mount_result = _mount("/sd", _vfs_open_sdcard());
    if (mount_result < 0) {
        printf("SD card mount failed: %d\n", mount_result);
        while(1);
    }
    printf("SD card mounted\n");

    uint32_t psram_base_addr = 0;
    int32_t bytes_loaded = load_file_to_psram("/sd/weights.bin", psram_base_addr);

    if (bytes_loaded < 0) {
        printf("Failed to load weights file\n");
        while(1);
    }

    printf("Weights loaded to PSRAM\n");

    load_weights_from_psram();

    memset((void *)&NN, 0, sizeof(NN));
    NN.flag_input = 0;
    NN.phase = 0;
    for (int i = 0; i < N_WORKERS; ++i) NN.worker_done[i] = 0;

    int cid_in = _cogstart_C(cog_input, &NN, stack_input, sizeof(stack_input));
    static WorkerParam wparams[N_WORKERS];
    int cid_workers[N_WORKERS];
    for (int i = 0; i < N_WORKERS; ++i) {
        wparams[i].shared = &NN;
        wparams[i].id = i;
        cid_workers[i] = _cogstart_C(cog_worker, &wparams[i], stacks_workers[i], sizeof(stacks_workers[i]));
    }

    printf("\nStarting inference\n");

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

    NN.phase = 4;
    for (int i = 0; i < N_WORKERS; ++i) NN.worker_done[i] = 0;
    do {
        done = 1;
        for (int i = 0; i < N_WORKERS; ++i) if (NN.worker_done[i] != 4) { done = 0; break; }
        if (!done) _waitx(1);
    } while (!done);
    NN.phase = 0;

    uint64_t ms_end = _getms();
    float ms_total = (float)(ms_end - ms_start);

    float logits[FC2_OUT];
    for (int k = 0; k < FC2_OUT; ++k) logits[k] = NN.out[k];
    softmax_f(logits, FC2_OUT);

    int argmax = 0;
    for (int i = 1; i < FC2_OUT; ++i) if (logits[i] > logits[argmax]) argmax = i;

    printf("\nPredicted digit: %d\n", argmax);
    printf("Probabilities:\n");
    for (int i = 0; i < FC2_OUT; ++i) printf("  %d: %.4f\n", i, logits[i]);
    printf("Inference time: %.3f ms\n", ms_total);

    _cogstop(cid_in);
    for (int i = 0; i < N_WORKERS; ++i) _cogstop(cid_workers[i]);

    for (;;) _waitx(1000);
    return 0;
}
