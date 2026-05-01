// =============================================================================
// bench_par.c  --  CNN Layer Parallelization Benchmark
// =============================================================================
// Sweeps num_workers (1..MAX_WORKERS) and partition axis (channel / row) for
// each layer independently to find the optimal LAYER_PROFILE configuration.
//
// Design differences vs 38kb_par.c:
//   1. SharedNN gains a `logical_phase` field so workers always know which
//      handler to call, even when the same logical phase runs multiple times
//      (the monotonic `phase` counter avoids re-triggering stale phases).
//   2. `cog_worker` spins on counter change, reads `logical_phase` for dispatch.
//   3. `main` runs a full benchmark sweep then prints a recommended profile
//      and timing comparison against the original hardcoded config.
// =============================================================================

enum { _clkfreq = 320000000 };
#include "38kb_weights.h"
#include "mnist_sample.h"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
#define MODEL_NAME    "38kb"
#define N_RUNS        10        // inference passes averaged per config
#define STACKSIZE     4096
#define WBUF_MAX_SIZE 72
#define _CLKFREQ      320000000

// Input
#define IN_H 28
#define IN_W 28
#define IN_C 1

// Conv1: 4 output channels, 3x3 kernel, same-pad, stride 1 -> 28x28
#define C1_OUT    4
#define C1_K      3
#define C1_H      28
#define C1_W      28
#define C1_STRIDE 1
#define PAD1_H    1
#define PAD1_W    1

// Conv2: 8 output channels, 3x3 kernel, no-pad, stride 2 -> 14x14
#define C2_OUT    8
#define C2_K      3
#define C2_H      14
#define C2_W      14
#define C2_STRIDE 2
#define PAD2_H    0
#define PAD2_W    0

// Conv3: 8 output channels, 3x3 kernel, same-pad, stride 1 -> 14x14
#define C3_OUT    8
#define C3_K      3
#define C3_H      14
#define C3_W      14
#define C3_STRIDE 1
#define PAD3_H    1
#define PAD3_W    1

// Conv4: 12 output channels, 3x3 kernel, no-pad, stride 2 -> 7x7
#define C4_OUT    12
#define C4_K      3
#define C4_H      7
#define C4_W      7
#define C4_STRIDE 2
#define PAD4_H    0
#define PAD4_W    0

#define GAP_OUT   12
#define DENSE_OUT 10

#define MAX_WORKERS 7

// Weight layout macros
#define IDX_CONV1_W(ky,kx,ic,oc) ((((ky)*C1_K+(kx))*IN_C+(ic))*C1_OUT+(oc))
#define IDX_CONV2_W(ky,kx,ic,oc) ((((ky)*C2_K+(kx))*C1_OUT+(ic))*C2_OUT+(oc))
#define IDX_CONV3_W(ky,kx,ic,oc) ((((ky)*C3_K+(kx))*C2_OUT+(ic))*C3_OUT+(oc))
#define IDX_CONV4_W(ky,kx,ic,oc) ((((ky)*C4_K+(kx))*C3_OUT+(ic))*C4_OUT+(oc))
#define IDX_DENSE_W(i,o)          ((i)*DENSE_OUT+(o))

#include <stdio.h>
#include <propeller2.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Shared memory layout
// ---------------------------------------------------------------------------
typedef struct {
    volatile float img[IN_H][IN_W];
    volatile float c1_out[C1_OUT * C1_H * C1_W];
    volatile float c2_out[C2_OUT * C2_H * C2_W];
    volatile float c3_out[C3_OUT * C3_H * C3_W];
    volatile float c4_out[C4_OUT * C4_H * C4_W];
    volatile float gap[GAP_OUT];
    volatile float out[DENSE_OUT];

    // Synchronization
    volatile int phase;              // Monotonically increasing counter
    volatile int logical_phase;      // 1-6: which handler to call
    volatile int active_workers;
    volatile int worker_done[MAX_WORKERS];
    volatile int layer_start[MAX_WORKERS];
    volatile int layer_count[MAX_WORKERS];
    volatile int partition_axis;     // 0=channel, 1=row
} SharedNN;

SharedNN NN;

typedef struct {
    SharedNN *shared;
    int id;
} WorkerParam;

static unsigned char stacks_workers[MAX_WORKERS][STACKSIZE];

// ---------------------------------------------------------------------------
// Layer profile type
// ---------------------------------------------------------------------------
typedef struct {
    int num_workers;
    int strategy;    // 1 = partition by output channel, 2 = partition by row
} LayerDesc;

// Original hardcoded profile (baseline for comparison)
static const LayerDesc ORIGINAL_PROFILE[6] = {
    {4, 1},  // Conv1
    {7, 1},  // Conv2
    {7, 1},  // Conv3
    {6, 1},  // Conv4
    {3, 1},  // GAP
    {2, 1},  // Dense
};

// Benchmark results: [phase 0-5][nw 0-6][strategy 0-1]
// strategy index 0 = channel, 1 = row. Value 0 means "not tested".
static uint32_t bench_us[6][MAX_WORKERS][2];

// Best config found per phase
static int best_nw[6];
static int best_strategy[6];  // 1 or 2

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------
static inline float relu_f(float x) { return x < 0.0f ? 0.0f : x; }

static void softmax_f(float *x, int len) {
    float maxv = x[0];
    for (int i = 1; i < len; ++i) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) { x[i] = expf(x[i] - maxv); sum += x[i]; }
    for (int i = 0; i < len; ++i) x[i] /= sum;
}

static inline void get_worker_range(int total, int id, int nw, int *start, int *count) {
    int base = total / nw;
    int rem  = total % nw;
    *start = id * base + (id < rem ? id : rem);
    *count = base + (id < rem ? 1 : 0);
}

// ---------------------------------------------------------------------------
// Layer handlers (identical to 38kb_par.c)
// ---------------------------------------------------------------------------

// --- Conv1 ---
static void handle_conv1_chan(SharedNN *s, int start, int count, float *wbuf) {
    for (int j = 0; j < count; j++) {
        int oc = start + j;
        float local_bias = conv2d_biases[oc];
        for (int ky = 0; ky < C1_K; ky++)
            for (int kx = 0; kx < C1_K; kx++)
                wbuf[ky*C1_K+kx] = conv2d_weights[IDX_CONV1_W(ky,kx,0,oc)];
        for (int oy = 0; oy < C1_H; ++oy)
            for (int ox = 0; ox < C1_W; ++ox) {
                float sum = local_bias;
                for (int ky = 0; ky < C1_K; ky++) {
                    int iy = oy + ky - PAD1_H;
                    for (int kx = 0; kx < C1_K; kx++) {
                        int ix = ox + kx - PAD1_W;
                        if (iy >= 0 && iy < IN_H && ix >= 0 && ix < IN_W)
                            sum += s->img[iy][ix] * wbuf[ky*C1_K+kx];
                    }
                }
                s->c1_out[(oc*C1_H+oy)*C1_W+ox] = relu_f(sum);
            }
    }
}

static void handle_conv1_row(SharedNN *s, int start, int count, float *wbuf) {
    for (int oc = 0; oc < C1_OUT; oc++) {
        float local_bias = conv2d_biases[oc];
        for (int ky = 0; ky < C1_K; ky++)
            for (int kx = 0; kx < C1_K; kx++)
                wbuf[ky*C1_K+kx] = conv2d_weights[IDX_CONV1_W(ky,kx,0,oc)];
        for (int oy = start; oy < start+count; ++oy)
            for (int ox = 0; ox < C1_W; ++ox) {
                float sum = local_bias;
                for (int ky = 0; ky < C1_K; ky++) {
                    int iy = oy + ky - PAD1_H;
                    for (int kx = 0; kx < C1_K; kx++) {
                        int ix = ox + kx - PAD1_W;
                        if (iy >= 0 && iy < IN_H && ix >= 0 && ix < IN_W)
                            sum += s->img[iy][ix] * wbuf[ky*C1_K+kx];
                    }
                }
                s->c1_out[(oc*C1_H+oy)*C1_W+ox] = relu_f(sum);
            }
    }
}

// --- Conv2 ---
static void handle_conv2_chan(SharedNN *s, int start, int count, float *wbuf) {
    for (int j = 0; j < count; j++) {
        int oc = start + j;
        float local_bias = conv2d_1_biases[oc];
        for (int ic = 0; ic < C1_OUT; ic++)
            for (int ky = 0; ky < C2_K; ky++)
                for (int kx = 0; kx < C2_K; kx++)
                    wbuf[ic*C2_K*C2_K + ky*C2_K + kx] =
                        conv2d_1_weights[IDX_CONV2_W(ky,kx,ic,oc)];
        for (int oy = 0; oy < C2_H; ++oy)
            for (int ox = 0; ox < C2_W; ++ox) {
                float sum = local_bias;
                for (int ic = 0; ic < C1_OUT; ic++) {
                    float *pw = &wbuf[ic*C2_K*C2_K];
                    for (int ky = 0; ky < C2_K; ky++) {
                        int iy = oy*C2_STRIDE + ky - PAD2_H;
                        for (int kx = 0; kx < C2_K; kx++) {
                            int ix = ox*C2_STRIDE + kx - PAD2_W;
                            if (iy >= 0 && iy < C1_H && ix >= 0 && ix < C1_W)
                                sum += s->c1_out[(ic*C1_H+iy)*C1_W+ix] * pw[ky*C2_K+kx];
                        }
                    }
                }
                s->c2_out[(oc*C2_H+oy)*C2_W+ox] = relu_f(sum);
            }
    }
}

static void handle_conv2_row(SharedNN *s, int start, int count, float *wbuf) {
    for (int oc = 0; oc < C2_OUT; oc++) {
        float local_bias = conv2d_1_biases[oc];
        for (int ic = 0; ic < C1_OUT; ic++)
            for (int ky = 0; ky < C2_K; ky++)
                for (int kx = 0; kx < C2_K; kx++)
                    wbuf[ic*C2_K*C2_K + ky*C2_K + kx] =
                        conv2d_1_weights[IDX_CONV2_W(ky,kx,ic,oc)];
        for (int oy = start; oy < start+count; ++oy)
            for (int ox = 0; ox < C2_W; ++ox) {
                float sum = local_bias;
                for (int ic = 0; ic < C1_OUT; ic++) {
                    float *pw = &wbuf[ic*C2_K*C2_K];
                    for (int ky = 0; ky < C2_K; ky++) {
                        int iy = oy*C2_STRIDE + ky - PAD2_H;
                        for (int kx = 0; kx < C2_K; kx++) {
                            int ix = ox*C2_STRIDE + kx - PAD2_W;
                            if (iy >= 0 && iy < C1_H && ix >= 0 && ix < C1_W)
                                sum += s->c1_out[(ic*C1_H+iy)*C1_W+ix] * pw[ky*C2_K+kx];
                        }
                    }
                }
                s->c2_out[(oc*C2_H+oy)*C2_W+ox] = relu_f(sum);
            }
    }
}

// --- Conv3 ---
static void handle_conv3_chan(SharedNN *s, int start, int count, float *wbuf) {
    for (int j = 0; j < count; j++) {
        int oc = start + j;
        float local_bias = conv2d_2_biases[oc];
        for (int ic = 0; ic < C2_OUT; ic++)
            for (int ky = 0; ky < C3_K; ky++)
                for (int kx = 0; kx < C3_K; kx++)
                    wbuf[ic*C3_K*C3_K + ky*C3_K + kx] =
                        conv2d_2_weights[IDX_CONV3_W(ky,kx,ic,oc)];
        for (int oy = 0; oy < C3_H; ++oy)
            for (int ox = 0; ox < C3_W; ++ox) {
                float sum = local_bias;
                for (int ic = 0; ic < C2_OUT; ic++) {
                    float *pw = &wbuf[ic*C3_K*C3_K];
                    for (int ky = 0; ky < C3_K; ky++) {
                        int iy = oy + ky - PAD3_H;
                        for (int kx = 0; kx < C3_K; kx++) {
                            int ix = ox + kx - PAD3_W;
                            if (iy >= 0 && iy < C2_H && ix >= 0 && ix < C2_W)
                                sum += s->c2_out[(ic*C2_H+iy)*C2_W+ix] * pw[ky*C3_K+kx];
                        }
                    }
                }
                s->c3_out[(oc*C3_H+oy)*C3_W+ox] = relu_f(sum);
            }
    }
}

static void handle_conv3_row(SharedNN *s, int start, int count, float *wbuf) {
    for (int oc = 0; oc < C3_OUT; oc++) {
        float local_bias = conv2d_2_biases[oc];
        for (int ic = 0; ic < C2_OUT; ic++)
            for (int ky = 0; ky < C3_K; ky++)
                for (int kx = 0; kx < C3_K; kx++)
                    wbuf[ic*C3_K*C3_K + ky*C3_K + kx] =
                        conv2d_2_weights[IDX_CONV3_W(ky,kx,ic,oc)];
        for (int oy = start; oy < start+count; ++oy)
            for (int ox = 0; ox < C3_W; ++ox) {
                float sum = local_bias;
                for (int ic = 0; ic < C2_OUT; ic++) {
                    float *pw = &wbuf[ic*C3_K*C3_K];
                    for (int ky = 0; ky < C3_K; ky++) {
                        int iy = oy + ky - PAD3_H;
                        for (int kx = 0; kx < C3_K; kx++) {
                            int ix = ox + kx - PAD3_W;
                            if (iy >= 0 && iy < C2_H && ix >= 0 && ix < C2_W)
                                sum += s->c2_out[(ic*C2_H+iy)*C2_W+ix] * pw[ky*C3_K+kx];
                        }
                    }
                }
                s->c3_out[(oc*C3_H+oy)*C3_W+ox] = relu_f(sum);
            }
    }
}

// --- Conv4 ---
static void handle_conv4_chan(SharedNN *s, int start, int count, float *wbuf) {
    for (int j = 0; j < count; j++) {
        int oc = start + j;
        float local_bias = conv2d_3_biases[oc];
        for (int ic = 0; ic < C3_OUT; ic++)
            for (int ky = 0; ky < C4_K; ky++)
                for (int kx = 0; kx < C4_K; kx++)
                    wbuf[ic*C4_K*C4_K + ky*C4_K + kx] =
                        conv2d_3_weights[IDX_CONV4_W(ky,kx,ic,oc)];
        for (int oy = 0; oy < C4_H; ++oy)
            for (int ox = 0; ox < C4_W; ++ox) {
                float sum = local_bias;
                for (int ic = 0; ic < C3_OUT; ic++) {
                    float *pw = &wbuf[ic*C4_K*C4_K];
                    for (int ky = 0; ky < C4_K; ky++) {
                        int iy = oy*C4_STRIDE + ky - PAD4_H;
                        for (int kx = 0; kx < C4_K; kx++) {
                            int ix = ox*C4_STRIDE + kx - PAD4_W;
                            if (iy >= 0 && iy < C3_H && ix >= 0 && ix < C3_W)
                                sum += s->c3_out[(ic*C3_H+iy)*C3_W+ix] * pw[ky*C4_K+kx];
                        }
                    }
                }
                s->c4_out[(oc*C4_H+oy)*C4_W+ox] = relu_f(sum);
            }
    }
}

static void handle_conv4_row(SharedNN *s, int start, int count, float *wbuf) {
    for (int oc = 0; oc < C4_OUT; oc++) {
        float local_bias = conv2d_3_biases[oc];
        for (int ic = 0; ic < C3_OUT; ic++)
            for (int ky = 0; ky < C4_K; ky++)
                for (int kx = 0; kx < C4_K; kx++)
                    wbuf[ic*C4_K*C4_K + ky*C4_K + kx] =
                        conv2d_3_weights[IDX_CONV4_W(ky,kx,ic,oc)];
        for (int oy = start; oy < start+count; ++oy)
            for (int ox = 0; ox < C4_W; ++ox) {
                float sum = local_bias;
                for (int ic = 0; ic < C3_OUT; ic++) {
                    float *pw = &wbuf[ic*C4_K*C4_K];
                    for (int ky = 0; ky < C4_K; ky++) {
                        int iy = oy*C4_STRIDE + ky - PAD4_H;
                        for (int kx = 0; kx < C4_K; kx++) {
                            int ix = ox*C4_STRIDE + kx - PAD4_W;
                            if (iy >= 0 && iy < C3_H && ix >= 0 && ix < C3_W)
                                sum += s->c3_out[(ic*C3_H+iy)*C3_W+ix] * pw[ky*C4_K+kx];
                        }
                    }
                }
                s->c4_out[(oc*C4_H+oy)*C4_W+ox] = relu_f(sum);
            }
    }
}

// --- GAP ---
static void handle_gap(SharedNN *s, int start, int count) {
    for (int j = 0; j < count; j++) {
        int ch = start + j;
        float sum = 0.0f;
        for (int y = 0; y < C4_H; ++y)
            for (int x = 0; x < C4_W; ++x)
                sum += s->c4_out[(ch*C4_H+y)*C4_W+x];
        s->gap[ch] = sum / (C4_H * C4_W);
    }
}

// --- Dense ---
static void handle_dense(SharedNN *s, int start, int count, float *wbuf) {
    for (int j = 0; j < count; j++) {
        int o = start + j;
        float local_bias = dense_biases[o];
        for (int i = 0; i < GAP_OUT; i++)
            wbuf[i] = dense_weights[IDX_DENSE_W(i,o)];
        float sum = local_bias;
        for (int i = 0; i < GAP_OUT; i++)
            sum += s->gap[i] * wbuf[i];
        s->out[o] = sum;
    }
}

// ---------------------------------------------------------------------------
// Worker cog
// Uses monotonic `phase` counter to detect new work; reads `logical_phase`
// (1-6) to decide which handler to call, so the same logical phase can run
// multiple times during benchmarking without getting stuck.
// ---------------------------------------------------------------------------
void cog_worker(void *p) {
    WorkerParam *wp = (WorkerParam *)p;
    SharedNN *s = wp->shared;
    int id = wp->id;

    __attribute__((cog)) float wbuf[WBUF_MAX_SIZE];

    int last_phase = 0;

    while (1) {
        // Spin until the monotonic counter advances
        int cur;
        do {
            cur = s->phase;
            if (cur == -1) return;
            if (cur != last_phase) break;
            _waitx(_clockfreq() / 2000);
        } while (1);

        last_phase = cur;

        // Read what logical layer we should process
        int lph  = s->logical_phase;
        int axis = s->partition_axis;

        if (id < s->active_workers) {
            int start = s->layer_start[id];
            int count = s->layer_count[id];
            if (count > 0) {
                switch (lph) {
                    case 1:
                        if (axis == 0) handle_conv1_chan(s, start, count, wbuf);
                        else           handle_conv1_row (s, start, count, wbuf);
                        break;
                    case 2:
                        if (axis == 0) handle_conv2_chan(s, start, count, wbuf);
                        else           handle_conv2_row (s, start, count, wbuf);
                        break;
                    case 3:
                        if (axis == 0) handle_conv3_chan(s, start, count, wbuf);
                        else           handle_conv3_row (s, start, count, wbuf);
                        break;
                    case 4:
                        if (axis == 0) handle_conv4_chan(s, start, count, wbuf);
                        else           handle_conv4_row (s, start, count, wbuf);
                        break;
                    case 5:
                        handle_gap  (s, start, count);
                        break;
                    case 6:
                        handle_dense(s, start, count, wbuf);
                        break;
                }
            }
        }

        s->worker_done[id] = cur;
    }
}

// ---------------------------------------------------------------------------
// Supervisor helpers
// ---------------------------------------------------------------------------

static int g_phase_counter = 0;  // monotonically increasing

// Returns the hard ceiling on useful workers for (logical_ph, strategy).
// Workers beyond this threshold receive count=0 and are effectively idle.
static int max_useful(int lph, int strategy) {
    if (strategy == 1) {   // channel axis
        switch (lph) {
            case 1: return C1_OUT;    // 4
            case 2: return C2_OUT;    // 8
            case 3: return C3_OUT;    // 8
            case 4: return C4_OUT;    // 12 (but cap at MAX_WORKERS=7)
            case 5: return GAP_OUT;   // 12
            case 6: return DENSE_OUT; // 10
        }
    } else {               // row axis
        switch (lph) {
            case 1: return C1_H;      // 28
            case 2: return C2_H;      // 14
            case 3: return C3_H;      // 14
            case 4: return C4_H;      // 7
            default: return 1;        // GAP/Dense: no row axis
        }
    }
    return MAX_WORKERS;
}

// Dispatches one layer to workers, returns elapsed microseconds.
// measure=0: run but don't timestamp (used for setup passes).
static uint32_t dispatch_phase(int lph, int nw, int strategy, int measure) {
    int out_c = 0, out_h = 0;
    switch (lph) {
        case 1: out_c = C1_OUT;    out_h = C1_H; break;
        case 2: out_c = C2_OUT;    out_h = C2_H; break;
        case 3: out_c = C3_OUT;    out_h = C3_H; break;
        case 4: out_c = C4_OUT;    out_h = C4_H; break;
        case 5: out_c = GAP_OUT;   out_h = 1;    break;
        case 6: out_c = DENSE_OUT; out_h = 1;    break;
    }

    int total_units        = (strategy == 1) ? out_c : out_h;
    NN.partition_axis      = (strategy == 1) ? 0 : 1;
    NN.logical_phase       = lph;
    NN.active_workers      = nw;

    for (int i = 0; i < MAX_WORKERS; i++) {
        if (i < nw) {
            int lstart, lcnt;
            get_worker_range(total_units, i, nw, &lstart, &lcnt);
            NN.layer_start[i] = lstart;
            NN.layer_count[i] = lcnt;
        } else {
            NN.layer_start[i] = 0;
            NN.layer_count[i] = 0;
        }
    }

    g_phase_counter++;

    uint32_t t0 = measure ? _getus() : 0;
    NN.phase = g_phase_counter;   // kick all cogs

    // Wait for every active worker to acknowledge
    int done;
    do {
        done = 1;
        for (int i = 0; i < nw; i++)
            if (NN.worker_done[i] != g_phase_counter) { done = 0; break; }
        if (!done) _waitx(_clockfreq() / 2000);
    } while (!done);

    return measure ? (_getus() - t0) : 0;
}

// Run phases 1..(target_ph-1) with the original profile as warm-up.
// Because computation is deterministic, the intermediate buffers are valid
// for all subsequent runs of target_ph without re-warming.
static void warmup_preceding(int target_ph) {
    for (int ph = 1; ph < target_ph; ph++)
        dispatch_phase(ph,
                       ORIGINAL_PROFILE[ph-1].num_workers,
                       ORIGINAL_PROFILE[ph-1].strategy,
                       0 /*no measure*/);
}

// Benchmark one (phase, nw, strategy) configuration: N_RUNS passes, averaged.
static uint32_t bench_one(int lph, int nw, int strategy) {
    uint32_t total = 0;
    for (int r = 0; r < N_RUNS; r++)
        total += dispatch_phase(lph, nw, strategy, 1 /*measure*/);
    return total / N_RUNS;
}

// Run a complete 6-phase inference with the given profile; return total us.
static uint32_t run_full_timed(const LayerDesc *profile) {
    uint32_t t0 = _getus();
    for (int ph = 1; ph <= 6; ph++)
        dispatch_phase(ph, profile[ph-1].num_workers, profile[ph-1].strategy, 0);
    return _getus() - t0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(void) {
    printf("\n");
    printf("================================================================\n");
    printf("  CNN Layer Parallelization Benchmark  --  model: %s\n", MODEL_NAME);
    printf("================================================================\n");
    printf("Clock : %lu Hz\n", (uint32_t)_clockfreq());
    printf("N_RUNS: %d  (averaged per configuration)\n", N_RUNS);
    printf("Sample: index=%d  true_label=%d\n\n", SAMPLE_INDEX, SAMPLE_LABEL);

    // ---- Initialise shared state ----
    memset((void *)&NN, 0, sizeof(NN));
    NN.phase = 0;
    g_phase_counter = 0;
    for (int y = 0; y < IN_H; ++y)
        for (int x = 0; x < IN_W; ++x)
            NN.img[y][x] = mnist_sample[y][x];

    // ---- Spawn worker cogs ----
    static WorkerParam wparams[MAX_WORKERS];
    int cid[MAX_WORKERS];
    for (int i = 0; i < MAX_WORKERS; i++) {
        wparams[i].shared = &NN;
        wparams[i].id     = i;
        cid[i] = _cogstart_C(cog_worker, &wparams[i],
                             stacks_workers[i], STACKSIZE);
    }

    // ---- Layer metadata for pretty printing ----
    static const char *phase_names[6] = {
        "Conv1", "Conv2", "Conv3", "Conv4", "GAP", "Dense"
    };
    static const char *phase_dims[6] = {
        "4ch x 28x28  in=1ch",
        "8ch x 14x14  stride=2",
        "8ch x 14x14  stride=1",
        "12ch x 7x7   stride=2",
        "12ch -> 12",
        "12 -> 10"
    };

    // ---- BENCHMARK SWEEP ----
    // For each layer:
    //   1. Warm up preceding layers once (deterministic, so valid for all runs).
    //   2. Sweep (strategy, nw) and record average time.
    //   3. Track best config.

    memset(bench_us, 0, sizeof(bench_us));

    for (int target_ph = 1; target_ph <= 6; target_ph++) {
        int idx = target_ph - 1;

        printf("----------------------------------------------------------------\n");
        printf("Phase %d  (%s)  --  %s\n",
               target_ph, phase_names[idx], phase_dims[idx]);
        printf("%-8s  %-3s  %8s  %8s  Notes\n",
               "Axis", "NW", "Avg(us)", "Speedup");

        // Warm up: run layers before this one once
        warmup_preceding(target_ph);

        // GAP and Dense have no meaningful row axis
        int max_strat = (target_ph <= 4) ? 2 : 1;

        best_nw[idx]       = 1;
        best_strategy[idx] = 1;
        uint32_t best_us   = UINT32_MAX;

        for (int strat = 1; strat <= max_strat; strat++) {
            const char *ax_name = (strat == 1) ? "channel" : "row";
            int ceil_nw = max_useful(target_ph, strat);
            uint32_t baseline = 0;

            for (int nw = 1; nw <= MAX_WORKERS; nw++) {
                uint32_t us = bench_one(target_ph, nw, strat);
                bench_us[idx][nw-1][strat-1] = us;

                if (nw == 1) baseline = us;

                // Effective speedup vs single-worker, same axis
                float spd = (baseline > 0) ? (float)baseline / (float)us : 1.0f;

                // Note: past the channel ceiling more workers don't divide work
                const char *note = (nw > ceil_nw) ? "[>ceil]" : "";
                // Mark if this is the overall best so far
                const char *best_mark = "";
                if (us < best_us) {
                    best_us           = us;
                    best_nw[idx]      = nw;
                    best_strategy[idx]= strat;
                    best_mark = " <--";
                }

                printf("%-8s  %-3d  %8lu  %7.2fx  %s%s\n",
                       ax_name, nw, us, spd, note, best_mark);
            }
            // Blank line between axis groups
            if (strat < max_strat) printf("\n");
        }

        printf("  >> BEST: %s axis, %d worker(s), %lu us\n\n",
               best_strategy[idx] == 1 ? "channel" : "row",
               best_nw[idx], best_us);
    }

    // ---- SUMMARY TABLE ----
    printf("================================================================\n");
    printf("  LAYER SUMMARY\n");
    printf("================================================================\n");
    printf("%-6s  %-6s  %-6s  %-9s  %-6s  %-6s  %-9s  %s\n",
           "Layer", "OrgNW", "OrgAx", "Org(us)",
           "OptNW", "OptAx", "Opt(us)", "Gain");
    printf("------  ------  ------  ---------  ------  ------  ---------  ----\n");

    for (int i = 0; i < 6; i++) {
        int o_nw   = ORIGINAL_PROFILE[i].num_workers;
        int o_strat= ORIGINAL_PROFILE[i].strategy;
        uint32_t o_us = bench_us[i][o_nw-1][o_strat-1];
        uint32_t b_us = bench_us[i][best_nw[i]-1][best_strategy[i]-1];
        float gain = (b_us > 0) ? (float)o_us / (float)b_us : 1.0f;

        printf("%-6s  %-6d  %-6s  %-9lu  %-6d  %-6s  %-9lu  %.2fx\n",
               phase_names[i],
               o_nw,   (o_strat==1) ? "chan" : "row",  o_us,
               best_nw[i], (best_strategy[i]==1) ? "chan" : "row", b_us,
               gain);
    }
    printf("\n");

    // ---- GENERATED OPTIMAL PROFILE ----
    printf("================================================================\n");
    printf("  RECOMMENDED LAYER_PROFILE  (paste into your inference program)\n");
    printf("================================================================\n");
    printf("static const LayerDesc LAYER_PROFILE[6] = {\n");
    for (int i = 0; i < 6; i++) {
        printf("    {%d, %d},  // %s  (%s axis)\n",
               best_nw[i], best_strategy[i],
               phase_names[i],
               best_strategy[i] == 1 ? "channel" : "row");
    }
    printf("};\n\n");

    // ---- FULL INFERENCE TIMING COMPARISON ----
    // Build optimal profile array from sweep results
    LayerDesc optimal_profile[6];
    for (int i = 0; i < 6; i++) {
        optimal_profile[i].num_workers = best_nw[i];
        optimal_profile[i].strategy    = best_strategy[i];
    }

    // Average N_RUNS full inferences for each profile
    printf("================================================================\n");
    printf("  FULL INFERENCE TIMING COMPARISON  (%d runs each)\n", N_RUNS);
    printf("================================================================\n");

    uint32_t orig_total = 0, opt_total = 0;

    for (int r = 0; r < N_RUNS; r++) {
        // Reset intermediate buffers between full runs
        memset((void*)NN.c1_out, 0, sizeof(NN.c1_out));
        memset((void*)NN.c2_out, 0, sizeof(NN.c2_out));
        memset((void*)NN.c3_out, 0, sizeof(NN.c3_out));
        memset((void*)NN.c4_out, 0, sizeof(NN.c4_out));
        memset((void*)NN.gap,    0, sizeof(NN.gap));
        memset((void*)NN.out,    0, sizeof(NN.out));
        orig_total += run_full_timed(ORIGINAL_PROFILE);
    }
    orig_total /= N_RUNS;

    for (int r = 0; r < N_RUNS; r++) {
        memset((void*)NN.c1_out, 0, sizeof(NN.c1_out));
        memset((void*)NN.c2_out, 0, sizeof(NN.c2_out));
        memset((void*)NN.c3_out, 0, sizeof(NN.c3_out));
        memset((void*)NN.c4_out, 0, sizeof(NN.c4_out));
        memset((void*)NN.gap,    0, sizeof(NN.gap));
        memset((void*)NN.out,    0, sizeof(NN.out));
        opt_total += run_full_timed(optimal_profile);
    }
    opt_total /= N_RUNS;

    float overall_gain = (opt_total > 0) ? (float)orig_total / (float)opt_total : 1.0f;

    printf("Original profile  : %lu us  (%lu.%01lu ms)\n",
           orig_total, orig_total/1000, orig_total%1000);
    printf("Optimal profile   : %lu us  (%lu.%01lu ms)\n",
           opt_total,  opt_total/1000,  opt_total%1000);
    printf("Overall speedup   : %.2fx\n\n", overall_gain);

    // Verify correctness with the optimal profile (one final inference)
    memset((void*)NN.c1_out, 0, sizeof(NN.c1_out));
    memset((void*)NN.c2_out, 0, sizeof(NN.c2_out));
    memset((void*)NN.c3_out, 0, sizeof(NN.c3_out));
    memset((void*)NN.c4_out, 0, sizeof(NN.c4_out));
    memset((void*)NN.gap,    0, sizeof(NN.gap));
    memset((void*)NN.out,    0, sizeof(NN.out));
    run_full_timed(optimal_profile);

    float logits[DENSE_OUT];
    for (int k = 0; k < DENSE_OUT; k++) logits[k] = NN.out[k];
    softmax_f(logits, DENSE_OUT);
    int pred = 0;
    for (int i = 1; i < DENSE_OUT; i++) if (logits[i] > logits[pred]) pred = i;

    printf("================================================================\n");
    printf("  CORRECTNESS CHECK (optimal profile)\n");
    printf("================================================================\n");
    for (int i = 0; i < DENSE_OUT; i++) {
        printf("  %d: %.6f", i, logits[i]);
        if (i == pred)         printf("  <-- PREDICTION");
        if (i == SAMPLE_LABEL) printf("  (TRUE LABEL)");
        printf("\n");
    }
    printf("\nPredicted: %d  |  True: %d  |  Correct: %s\n\n",
           pred, SAMPLE_LABEL, (pred == SAMPLE_LABEL) ? "YES" : "NO");

    // ---- Shutdown ----
    NN.phase = -1;
    _waitx(_clockfreq() / 100);
    for (int i = 0; i < MAX_WORKERS; i++) _cogstop(cid[i]);

    for (;;) _waitx(_clockfreq() / 2000);
}
