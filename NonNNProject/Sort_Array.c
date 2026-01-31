#include <propeller2.h>
#include <stdio.h>
#include <stdint.h>

#define NUM_COGS 7
#define STACK_SIZE 2048
#define MICRO_CHUNK 64
#define MAX_ARRAY 5000
#define MAX_REGION 2000
#define RUNS 64

unsigned char stacks[NUM_COGS][STACK_SIZE];

typedef struct {
    int *array;
    int start;
    int end;
    int *buffer;
    volatile int done;
} work_t;

work_t work[NUM_COGS];
int cog_ids[NUM_COGS];

int local_buffers[NUM_COGS][MAX_REGION];
int final_buffer[MAX_ARRAY];

void insertion_sort(int *a, int n) {
    for (int i = 1; i < n; i++) {
        int v = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > v) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = v;
    }
}

void merge_local(int *a, int l, int m, int r, int *b) {
    int i = l, j = m, k = 0;
    while (i < m && j < r) b[k++] = (a[i] <= a[j]) ? a[i++] : a[j++];
    while (i < m) b[k++] = a[i++];
    while (j < r) b[k++] = a[j++];
    for (int x = 0; x < k; x++) a[l + x] = b[x];
}

void merge_final(int *a, int l, int m, int r) {
    int i = l, j = m, k = 0;
    while (i < m && j < r) final_buffer[k++] = (a[i] <= a[j]) ? a[i++] : a[j++];
    while (i < m) final_buffer[k++] = a[i++];
    while (j < r) final_buffer[k++] = a[j++];
    for (int x = 0; x < k; x++) a[l + x] = final_buffer[x];
}

void cog_sort(void *p) {
    work_t *w = (work_t *)p;
    int temp[MICRO_CHUNK];

    for (int pos = w->start; pos < w->end; pos += MICRO_CHUNK) {
        int n = (pos + MICRO_CHUNK > w->end) ? (w->end - pos) : MICRO_CHUNK;
        for (int i = 0; i < n; i++) temp[i] = w->array[pos + i];
        insertion_sort(temp, n);
        for (int i = 0; i < n; i++) w->array[pos + i] = temp[i];
    }

    int size = w->end - w->start;
    for (int step = MICRO_CHUNK; step < size; step <<= 1) {
        for (int pos = w->start; pos < w->end; pos += step << 1) {
            int l = pos;
            int m = pos + step;
            int r = pos + (step << 1);
            if (m >= w->end) break;
            if (r > w->end) r = w->end;
            merge_local(w->array, l, m, r, w->buffer);
        }
    }

    w->done = 1;
}

uint32_t seq_time_ms(int *a, int n) {
    uint32_t t0 = _getms();
    insertion_sort(a, n);
    uint32_t t1 = _getms();
    return (uint32_t)(t1 - t0);
}

uint32_t par_time_ms(int *a, int n) {
    int base = n / NUM_COGS;

    for (int i = 0; i < NUM_COGS; i++) {
        work[i].array = a;
        work[i].start = i * base;
        work[i].end = (i == NUM_COGS - 1) ? n : work[i].start + base;
        work[i].buffer = local_buffers[i];
        work[i].done = 0;
        cog_ids[i] = _cogstart_C(cog_sort, &work[i], stacks[i], STACK_SIZE);
    }

    uint32_t t0 = _getms();

    int done;
    do {
        done = 1;
        for (int i = 0; i < NUM_COGS; i++) {
            if (!work[i].done) done = 0;
        }
    } while (!done);

    for (int i = 0; i < NUM_COGS; i++) _cogstop(cog_ids[i]);

    for (int step = base; step < n; step <<= 1) {
        for (int pos = 0; pos < n; pos += step << 1) {
            int l = pos;
            int m = pos + step;
            int r = pos + (step << 1);
            if (m >= n) break;
            if (r > n) r = n;
            merge_final(a, l, m, r);
        }
    }

    uint32_t t1 = _getms();
    return (uint32_t)(t1 - t0);
}

int verify(int *a, int n) {
    for (int i = 0; i < n - 1; i++) if (a[i] > a[i + 1]) return 0;
    return 1;
}

int A[MAX_ARRAY];
int B[MAX_ARRAY];

int main(void) {
    int sizes[] = {210, 1050, 2100, 4200};
    const char *labels[] = {"Small", "Medium", "Large", "Larger"};

    for (int t = 0; t < 4; t++) {
        int n = sizes[t];
        uint64_t seq = 0;
        uint64_t par = 0;

        for (int r = 0; r < RUNS; r++) {
            unsigned int seed = 12345;
            for (int i = 0; i < n; i++) {
                seed = seed * 1103515245 + 12345;
                int v = (seed >> 16) & 0x3FFF;
                A[i] = v;
                B[i] = v;
            }

            seq += seq_time_ms(A, n);
            par += par_time_ms(B, n);
        }

        float fs = (float)seq / RUNS;
        float fp = (float)par / RUNS;
        float speedup = fs / fp;

        int ok = verify(A, n) && verify(B, n);

        printf("%s (%d): Seq %.2f ms  Par %.2f ms  %.2fx  \n",
               labels[t], n, fs, fp, speedup);
    }

    while (1) _waitx(_clockfreq());
    return 0;
}
