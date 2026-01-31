#include <propeller2.h>
#include <stdint.h>
#include <stdio.h>

#define NUM_COGS 7
#define STACK_SIZE 512
#define ITERATIONS 1000
#define MATRIX_SIZE 7
#define TOTAL_ELEMENTS 49

unsigned char stacks[NUM_COGS][STACK_SIZE];

typedef struct {
    int row;
    int *A;
    int *B;
    int *C;
    volatile int done;
} work_t;

work_t work[NUM_COGS];
int cog_ids[NUM_COGS];

int A[TOTAL_ELEMENTS];
int B[TOTAL_ELEMENTS];
int C_seq[TOTAL_ELEMENTS];
int C_par[TOTAL_ELEMENTS];

void cog_multiply(void *p) {
    work_t *w = (work_t *)p;
    int row = w->row;
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int col = 0; col < MATRIX_SIZE; col++) {
            int sum = 0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                sum += w->A[row * MATRIX_SIZE + k] * w->B[k * MATRIX_SIZE + col];
            }
            w->C[row * MATRIX_SIZE + col] = sum;
        }
    }
    
    w->done = 1;
}

uint32_t seq_time_ms() {
    uint32_t t0 = _getus();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                int sum = 0;
                for (int k = 0; k < MATRIX_SIZE; k++) {
                    sum += A[i * MATRIX_SIZE + k] * B[k * MATRIX_SIZE + j];
                }
                C_seq[i * MATRIX_SIZE + j] = sum;
            }
        }
    }
    
    uint32_t t1 = _getus();
    return (uint32_t)(t1 - t0);
}

uint32_t par_time_ms() {
    uint32_t t0 = _getus();  
    
    for (int i = 0; i < NUM_COGS; i++) {
        work[i].row = i;
        work[i].A = A;
        work[i].B = B;
        work[i].C = C_par;
        work[i].done = 0;
        
        cog_ids[i] = _cogstart_C(cog_multiply, &work[i], stacks[i], STACK_SIZE);
    }
    
    int done;
    do {
        done = 1;
        for (int i = 0; i < NUM_COGS; i++) {
            if (!work[i].done) done = 0;
        }
    } while (!done);
    
    for (int i = 0; i < NUM_COGS; i++) _cogstop(cog_ids[i]);
    
    uint32_t t1 = _getus();
    return (uint32_t)(t1 - t0);
}

int verify(int *C1, int *C2) {
    for (int i = 0; i < TOTAL_ELEMENTS; i++) {
        if (C1[i] != C2[i]) {
            printf("Mismatch at index %d: %d != %d\n", i, C1[i], C2[i]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    int counter = 1;
    for (int i = 0; i < TOTAL_ELEMENTS; i++) {
        A[i] = counter;
        B[i] = TOTAL_ELEMENTS - counter + 1;
        counter++;
        C_par[i] = 0;
        C_seq[i] = 0;
    }
    
    uint32_t seq_time = seq_time_ms();
    uint32_t par_time = par_time_ms();
    
    int ok = verify(C_seq, C_par);
    
    printf("Iterations: %d\n", ITERATIONS);
    printf("Matrix Size: %dx%d\n", MATRIX_SIZE, MATRIX_SIZE);
    printf("Cogs Used: %d\n", NUM_COGS);
    printf("Sequential: %lu us\n", seq_time);
    printf("Parallel: %lu us\n", par_time);
    printf("Results match: %s\n", ok ? "Yes" : "No");
    
    if (par_time > 0) {
        float speedup = (float)seq_time / (float)par_time;
        printf("Speedup: %.2fx\n", speedup);
    } else {
        printf("Speedup: N/A (parallel time too small)\n");
    }
    
    printf("\nMatrix A (7x7):\n");
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            printf("%3d ", A[i * MATRIX_SIZE + j]);
        }
        printf("\n");
    }
    
    printf("\nMatrix B (7x7):\n");
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            printf("%3d ", B[i * MATRIX_SIZE + j]);
        }
        printf("\n");
    }
    
    printf("\nResult (7x7):\n");
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            printf("%5d ", C_seq[i * MATRIX_SIZE + j]);
        }
        printf("\n");
    }
    
    while (1) _waitx(_clockfreq());
    return 0;
}
