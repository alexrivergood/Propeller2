// =============================================================================
// bench_105kb.c  --  CNN Layer Parallelization Benchmark  (105kb model)
// =============================================================================
// Sweeps num_workers (1..MAX_WORKERS) x partition axis (channel / row) for
// each layer independently to find the optimal LAYER_PROFILE.
//
// Worker architecture: generic conv2d_layer() / gap_layer() / dense_layer()
// functions (same arithmetic as the original, simpler dispatch).
// Monotonic phase counter + logical_phase field decouples "new work" from
// "what handler to call", allowing the same logical phase to run many times.
// =============================================================================

enum { _clkfreq = 320000000 };
#include "105kb_weights.h"
#include "mnist_sample.h"

// ---------------------------------------------------------------------------
// Model configuration
// ---------------------------------------------------------------------------
#define MODEL_NAME    "105kb"
#define N_RUNS        10          // inference passes averaged per config
#define STACKSIZE     4096
#define WBUF_MAX_SIZE 180
#define _CLKFREQ      320000000

#define IN_H 28
#define IN_W 28
#define IN_C 1

#define C1_OUT 10
#define C1_K   3
#define C1_H   28
#define C1_W   28
#define C1_STRIDE 1
#define PAD1_H 1
#define PAD1_W 1

#define C2_OUT 20
#define C2_K   3
#define C2_H   14
#define C2_W   14
#define C2_STRIDE 2
#define PAD2_H 0
#define PAD2_W 0

#define C3_OUT 20
#define C3_K   3
#define C3_H   14
#define C3_W   14
#define C3_STRIDE 1
#define PAD3_H 1
#define PAD3_W 1

#define C4_OUT 20
#define C4_K   3
#define C4_H   7
#define C4_W   7
#define C4_STRIDE 2
#define PAD4_H 0
#define PAD4_W 0

#define GAP_OUT   20
#define DENSE_OUT 10
#define MAX_WORKERS 7

#include <stdio.h>
#include <propeller2.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Shared memory
// ---------------------------------------------------------------------------
typedef struct {
    volatile float img[IN_H][IN_W];
    volatile float c1_out[C1_OUT * C1_H * C1_W];
    volatile float c2_out[C2_OUT * C2_H * C2_W];
    volatile float c3_out[C3_OUT * C3_H * C3_W];
    volatile float c4_out[C4_OUT * C4_H * C4_W];
    volatile float gap[GAP_OUT];
    volatile float out[DENSE_OUT];

    volatile int phase;           // Monotonically increasing counter
    volatile int logical_phase;   // 1-6: which layer handler to call
    volatile int active_workers;
    volatile int worker_done[MAX_WORKERS];
    volatile int layer_start[MAX_WORKERS];
    volatile int layer_count[MAX_WORKERS];
    volatile int partition_axis;  // 0=channel, 1=row
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
    int strategy;   // 1=channel axis, 2=row axis
} LayerDesc;

// Original hardcoded profile (baseline)
static const LayerDesc ORIGINAL_PROFILE[6] = {
    {5, 1},  // Conv1
    {7, 1},  // Conv2
    {7, 1},  // Conv3
    {7, 1},  // Conv4
    {5, 1},  // GAP
    {2, 1},  // Dense
};

// Results: [phase 0-5][nw 0-6][strategy 0=chan / 1=row]
static uint32_t bench_us[6][MAX_WORKERS][2];
static int best_nw[6];
static int best_strategy[6];

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
    int base = total / nw, rem = total % nw;
    *start = id * base + (id < rem ? id : rem);
    *count = base + (id < rem ? 1 : 0);
}

// ---------------------------------------------------------------------------
// Generic layer functions
// ---------------------------------------------------------------------------
static void conv2d_layer(volatile float *input, volatile float *output,
                         int in_c, int in_h, int in_w,
                         int out_c, int out_h, int out_w,
                         int k, int stride, int pad_h, int pad_w,
                         const float *weights, const float *bias,
                         int start, int count, int axis, float *wbuf)
{
    if (axis == 0) {                    /* channel-wise partition */
        for (int j = 0; j < count; j++) {
            int oc = start + j;
            float lb = bias[oc];
            for (int ic = 0; ic < in_c; ic++)
                for (int ky = 0; ky < k; ky++)
                    for (int kx = 0; kx < k; kx++)
                        wbuf[ic*k*k + ky*k + kx] =
                            weights[((ky*k + kx)*in_c + ic)*out_c + oc];
            for (int oy = 0; oy < out_h; ++oy)
                for (int ox = 0; ox < out_w; ++ox) {
                    float sum = lb;
                    for (int ic = 0; ic < in_c; ic++) {
                        float *pw = &wbuf[ic*k*k];
                        for (int ky = 0; ky < k; ky++) {
                            int iy = oy*stride + ky - pad_h;
                            for (int kx = 0; kx < k; kx++) {
                                int ix = ox*stride + kx - pad_w;
                                if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w)
                                    sum += input[(ic*in_h + iy)*in_w + ix] * pw[ky*k + kx];
                            }
                        }
                    }
                    output[(oc*out_h + oy)*out_w + ox] = relu_f(sum);
                }
        }
    } else {                           /* row-wise partition */
        for (int oc = 0; oc < out_c; oc++) {
            float lb = bias[oc];
            for (int ic = 0; ic < in_c; ic++)
                for (int ky = 0; ky < k; ky++)
                    for (int kx = 0; kx < k; kx++)
                        wbuf[ic*k*k + ky*k + kx] =
                            weights[((ky*k + kx)*in_c + ic)*out_c + oc];
            for (int oy = start; oy < start + count; ++oy)
                for (int ox = 0; ox < out_w; ++ox) {
                    float sum = lb;
                    for (int ic = 0; ic < in_c; ic++) {
                        float *pw = &wbuf[ic*k*k];
                        for (int ky = 0; ky < k; ky++) {
                            int iy = oy*stride + ky - pad_h;
                            for (int kx = 0; kx < k; kx++) {
                                int ix = ox*stride + kx - pad_w;
                                if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w)
                                    sum += input[(ic*in_h + iy)*in_w + ix] * pw[ky*k + kx];
                            }
                        }
                    }
                    output[(oc*out_h + oy)*out_w + ox] = relu_f(sum);
                }
        }
    }
}

static void gap_layer(volatile float *input, volatile float *output,
                      int c, int h, int w, int start, int count)
{
    float sp = (float)(h * w);
    for (int j = 0; j < count; j++) {
        int ch = start + j;
        float sum = 0.0f;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                sum += input[(ch*h + y)*w + x];
        output[ch] = sum / sp;
    }
}

static void dense_layer(volatile float *input, volatile float *output,
                        int in_size, int out_size,
                        const float *weights, const float *bias,
                        int start, int count, float *wbuf)
{
    for (int j = 0; j < count; j++) {
        int o = start + j;
        float lb = bias[o];
        for (int i = 0; i < in_size; i++)
            wbuf[i] = weights[i*out_size + o];
        float sum = lb;
        for (int i = 0; i < in_size; i++)
            sum += input[i] * wbuf[i];
        output[o] = sum;
    }
}

// ---------------------------------------------------------------------------
// Worker cog
// ---------------------------------------------------------------------------
void cog_worker(void *p) {
    WorkerParam *wp = (WorkerParam *)p;
    SharedNN *s = wp->shared;
    int id = wp->id;

    #if WBUF_MAX_SIZE <= 512
    __attribute__((cog)) float wbuf[WBUF_MAX_SIZE];
    #else
    float wbuf[WBUF_MAX_SIZE];
    #endif

    int last_phase = 0;

    while (1) {
        int cur;
        do {
            cur = s->phase;
            if (cur == -1) return;
            if (cur != last_phase) break;
            _waitx(_clockfreq() / 2000);
        } while (1);

        last_phase = cur;
        int lph  = s->logical_phase;
        int axis = s->partition_axis;

        if (id < s->active_workers) {
            int start = s->layer_start[id];
            int count = s->layer_count[id];
            if (count > 0) {
                switch (lph) {
                    case 1:
                        conv2d_layer((volatile float*)s->img, s->c1_out,
                                     IN_C, IN_H, IN_W, C1_OUT, C1_H, C1_W,
                                     C1_K, C1_STRIDE, PAD1_H, PAD1_W,
                                     conv2d_weights, conv2d_biases,
                                     start, count, axis, wbuf);
                        break;
                    case 2:
                        conv2d_layer(s->c1_out, s->c2_out,
                                     C1_OUT, C1_H, C1_W, C2_OUT, C2_H, C2_W,
                                     C2_K, C2_STRIDE, PAD2_H, PAD2_W,
                                     conv2d_1_weights, conv2d_1_biases,
                                     start, count, axis, wbuf);
                        break;
                    case 3:
                        conv2d_layer(s->c2_out, s->c3_out,
                                     C2_OUT, C2_H, C2_W, C3_OUT, C3_H, C3_W,
                                     C3_K, C3_STRIDE, PAD3_H, PAD3_W,
                                     conv2d_2_weights, conv2d_2_biases,
                                     start, count, axis, wbuf);
                        break;
                    case 4:
                        conv2d_layer(s->c3_out, s->c4_out,
                                     C3_OUT, C3_H, C3_W, C4_OUT, C4_H, C4_W,
                                     C4_K, C4_STRIDE, PAD4_H, PAD4_W,
                                     conv2d_3_weights, conv2d_3_biases,
                                     start, count, axis, wbuf);
                        break;
                    case 5:
                        gap_layer(s->c4_out, s->gap,
                                  C4_OUT, C4_H, C4_W, start, count);
                        break;
                    case 6:
                        dense_layer(s->gap, s->out,
                                    GAP_OUT, DENSE_OUT,
                                    dense_weights, dense_biases,
                                    start, count, wbuf);
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
static int g_phase_counter = 0;

static int max_useful(int lph, int strategy) {
    if (strategy == 1) {
        switch (lph) {
            case 1: return C1_OUT;
            case 2: return C2_OUT;
            case 3: return C3_OUT;
            case 4: return C4_OUT;
            case 5: return GAP_OUT;
            case 6: return DENSE_OUT;
        }
    } else {
        switch (lph) {
            case 1: return C1_H;
            case 2: return C2_H;
            case 3: return C3_H;
            case 4: return C4_H;
            default: return 1;
        }
    }
    return MAX_WORKERS;
}

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
    int total_units   = (strategy == 1) ? out_c : out_h;
    NN.partition_axis = (strategy == 1) ? 0 : 1;
    NN.logical_phase  = lph;
    NN.active_workers = nw;

    for (int i = 0; i < MAX_WORKERS; i++) {
        if (i < nw) {
            int ls, lc;
            get_worker_range(total_units, i, nw, &ls, &lc);
            NN.layer_start[i] = ls;
            NN.layer_count[i] = lc;
        } else {
            NN.layer_start[i] = 0;
            NN.layer_count[i] = 0;
        }
    }

    g_phase_counter++;
    uint32_t t0 = measure ? _getus() : 0;
    NN.phase = g_phase_counter;

    int done;
    do {
        done = 1;
        for (int i = 0; i < nw; i++)
            if (NN.worker_done[i] != g_phase_counter) { done = 0; break; }
        if (!done) _waitx(_clockfreq() / 2000);
    } while (!done);

    return measure ? (_getus() - t0) : 0;
}

static void warmup_preceding(int target_ph) {
    for (int ph = 1; ph < target_ph; ph++)
        dispatch_phase(ph, ORIGINAL_PROFILE[ph-1].num_workers,
                       ORIGINAL_PROFILE[ph-1].strategy, 0);
}

static uint32_t bench_one(int lph, int nw, int strategy) {
    uint32_t total = 0;
    for (int r = 0; r < N_RUNS; r++)
        total += dispatch_phase(lph, nw, strategy, 1);
    return total / N_RUNS;
}

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

    memset((void *)&NN, 0, sizeof(NN));
    NN.phase = 0;
    g_phase_counter = 0;
    for (int y = 0; y < IN_H; ++y)
        for (int x = 0; x < IN_W; ++x)
            NN.img[y][x] = mnist_sample[y][x];

    static WorkerParam wparams[MAX_WORKERS];
    int cid[MAX_WORKERS];
    for (int i = 0; i < MAX_WORKERS; i++) {
        wparams[i].shared = &NN;
        wparams[i].id     = i;
        cid[i] = _cogstart_C(cog_worker, &wparams[i],
                             stacks_workers[i], STACKSIZE);
    }

    static const char *phase_names[6] = {
        "Conv1","Conv2","Conv3","Conv4","GAP","Dense"
    };
    static const char *phase_dims[6] = {
        "10ch x 28x28  in=1ch", "20ch x 14x14  stride=2", "20ch x 14x14  stride=1", "20ch x 7x7    stride=2", "20ch -> 20", "20 -> 10"
    };

    memset(bench_us, 0, sizeof(bench_us));

    for (int target_ph = 1; target_ph <= 6; target_ph++) {
        int idx = target_ph - 1;
        printf("----------------------------------------------------------------\n");
        printf("Phase %d  (%s)  --  %s\n",
               target_ph, phase_names[idx], phase_dims[idx]);
        printf("%-8s  %-3s  %8s  %8s  Notes\n", "Axis","NW","Avg(us)","Speedup");

        warmup_preceding(target_ph);

        int max_strat = (target_ph <= 4) ? 2 : 1;
        best_nw[idx]       = 1;
        best_strategy[idx] = 1;
        uint32_t best_us_val = (uint32_t)-1;

        for (int strat = 1; strat <= max_strat; strat++) {
            const char *ax = (strat == 1) ? "channel" : "row";
            int ceil_nw = max_useful(target_ph, strat);
            uint32_t baseline = 0;
            for (int nw = 1; nw <= MAX_WORKERS; nw++) {
                uint32_t us = bench_one(target_ph, nw, strat);
                bench_us[idx][nw-1][strat-1] = us;
                if (nw == 1) baseline = us;
                float spd = (baseline > 0) ? (float)baseline / (float)us : 1.0f;
                const char *note = (nw > ceil_nw) ? "[>ceil]" : "";
                const char *mark = "";
                if (us < best_us_val) {
                    best_us_val = us;
                    best_nw[idx] = nw;
                    best_strategy[idx] = strat;
                    mark = " <--";
                }
                printf("%-8s  %-3d  %8lu  %7.2fx  %s%s\n",
                       ax, nw, us, spd, note, mark);
            }
            if (strat < max_strat) printf("\n");
        }
        printf("  >> BEST: %s axis, %d worker(s), %lu us\n\n",
               best_strategy[idx]==1 ? "channel":"row",
               best_nw[idx], best_us_val);
    }

    printf("================================================================\n");
    printf("  LAYER SUMMARY\n");
    printf("================================================================\n");
    printf("%-6s  %-6s  %-6s  %-9s  %-6s  %-6s  %-9s  %s\n",
           "Layer","OrgNW","OrgAx","Org(us)","OptNW","OptAx","Opt(us)","Gain");
    printf("------  ------  ------  ---------  ------  ------  ---------  ----\n");
    for (int i = 0; i < 6; i++) {
        int o_nw = ORIGINAL_PROFILE[i].num_workers;
        int o_st = ORIGINAL_PROFILE[i].strategy;
        uint32_t o_us = bench_us[i][o_nw-1][o_st-1];
        uint32_t b_us = bench_us[i][best_nw[i]-1][best_strategy[i]-1];
        float gain = (b_us > 0) ? (float)o_us/(float)b_us : 1.0f;
        printf("%-6s  %-6d  %-6s  %-9lu  %-6d  %-6s  %-9lu  %.2fx\n",
               phase_names[i],
               o_nw, (o_st==1)?"chan":"row", o_us,
               best_nw[i], (best_strategy[i]==1)?"chan":"row", b_us,
               gain);
    }
    printf("\n");

    printf("================================================================\n");
    printf("  RECOMMENDED LAYER_PROFILE\n");
    printf("================================================================\n");
    printf("static const LayerDesc LAYER_PROFILE[6] = {\n");
    for (int i = 0; i < 6; i++)
        printf("    {%d, %d},  // %s (%s axis)\n",
               best_nw[i], best_strategy[i], phase_names[i],
               best_strategy[i]==1?"channel":"row");
    printf("};\n\n");

    LayerDesc optimal[6];
    for (int i = 0; i < 6; i++) {
        optimal[i].num_workers = best_nw[i];
        optimal[i].strategy    = best_strategy[i];
    }

    printf("================================================================\n");
    printf("  FULL INFERENCE TIMING COMPARISON  (%d runs each)\n", N_RUNS);
    printf("================================================================\n");

    uint32_t orig_total = 0, opt_total = 0;
    for (int r = 0; r < N_RUNS; r++) {
        memset((void*)NN.c1_out,0,sizeof(NN.c1_out));
        memset((void*)NN.c2_out,0,sizeof(NN.c2_out));
        memset((void*)NN.c3_out,0,sizeof(NN.c3_out));
        memset((void*)NN.c4_out,0,sizeof(NN.c4_out));
        memset((void*)NN.gap,0,sizeof(NN.gap));
        memset((void*)NN.out,0,sizeof(NN.out));
        orig_total += run_full_timed(ORIGINAL_PROFILE);
    }
    orig_total /= N_RUNS;
    for (int r = 0; r < N_RUNS; r++) {
        memset((void*)NN.c1_out,0,sizeof(NN.c1_out));
        memset((void*)NN.c2_out,0,sizeof(NN.c2_out));
        memset((void*)NN.c3_out,0,sizeof(NN.c3_out));
        memset((void*)NN.c4_out,0,sizeof(NN.c4_out));
        memset((void*)NN.gap,0,sizeof(NN.gap));
        memset((void*)NN.out,0,sizeof(NN.out));
        opt_total += run_full_timed(optimal);
    }
    opt_total /= N_RUNS;

    float gain = (opt_total > 0) ? (float)orig_total/(float)opt_total : 1.0f;
    printf("Original profile  : %lu us  (%lu.%01lu ms)\n",
           orig_total, orig_total/1000, orig_total%1000);
    printf("Optimal profile   : %lu us  (%lu.%01lu ms)\n",
           opt_total,  opt_total/1000,  opt_total%1000);
    printf("Overall speedup   : %.2fx\n\n", gain);

    // Correctness check with optimal profile
    memset((void*)NN.c1_out,0,sizeof(NN.c1_out));
    memset((void*)NN.c2_out,0,sizeof(NN.c2_out));
    memset((void*)NN.c3_out,0,sizeof(NN.c3_out));
    memset((void*)NN.c4_out,0,sizeof(NN.c4_out));
    memset((void*)NN.gap,0,sizeof(NN.gap));
    memset((void*)NN.out,0,sizeof(NN.out));
    run_full_timed(optimal);

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
           pred, SAMPLE_LABEL, (pred==SAMPLE_LABEL)?"YES":"NO");

    NN.phase = -1;
    _waitx(_clockfreq() / 100);
    for (int i = 0; i < MAX_WORKERS; i++) _cogstop(cid[i]);
    for (;;) _waitx(_clockfreq() / 2000);
}
