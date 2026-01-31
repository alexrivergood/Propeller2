#include <propeller2.h>
#include <stdint.h>
#include <stdio.h>

#define NUM_COGS 3
#define STACK_SIZE 512
#define ITERATIONS 1000

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

int A[9];
int B[9];
int C_seq[9];
int C_par[9];

void cog_multiply(void *p) {
    work_t *w = (work_t *)p;
    int row = w->row;
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int col = 0; col < 3; col++) {
            int sum = 0;
            for (int k = 0; k < 3; k++) {
                sum += w->A[row * 3 + k] * w->B[k * 3 + col];
            }
            w->C[row * 3 + col] = sum;
        }
    }
    
    w->done = 1;
}

uint32_t seq_time_ms() {
    uint32_t t0 = _getus();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int sum = 0;
                for (int k = 0; k < 3; k++) {
                    sum += A[i * 3 + k] * B[k * 3 + j];
                }
                C_seq[i * 3 + j] = sum;
            }
        }
    }
    
    uint32_t t1 = _getus();
    return (uint32_t)(t1 - t0);
}

uint32_t par_time_ms() {
    uint32_t t0 = _getus();  
    
    // Initialize work parameters for each cog
    for (int i = 0; i < NUM_COGS; i++) {
        work[i].row = i;          // Core 0 gets row 0, core 1 gets row 1, etc.
        work[i].A = A;            // Pointer to matrix A
        work[i].B = B;            // Pointer to matrix B
        work[i].C = C_par;        // Pointer to result matrix
        work[i].done = 0;         // Not done yet
        
        cog_ids[i] = _cogstart_C(cog_multiply, &work[i], stacks[i], STACK_SIZE);
    }
    
    // Busy-wait loop to wait for workers
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
    for (int i = 0; i < 9; i++) {
        if (C1[i] != C2[i]) {
            printf("Mismatch at index %d: %d != %d\n", i, C1[i], C2[i]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    A[0] = 1; A[1] = 2; A[2] = 3;
    A[3] = 4; A[4] = 5; A[5] = 6;
    A[6] = 7; A[7] = 8; A[8] = 9;
    
    B[0] = 9; B[1] = 8; B[2] = 7;
    B[3] = 6; B[4] = 5; B[5] = 4;
    B[6] = 3; B[7] = 2; B[8] = 1;
    
    // Initialize C_par to zeros (optional, but good practice)
    for (int i = 0; i < 9; i++) {
        C_par[i] = 0;
        C_seq[i] = 0;
    }
    
    uint32_t seq_time = seq_time_ms();
    uint32_t par_time = par_time_ms();
    
    int ok = verify(C_seq, C_par);
    
    printf("Iterations: %d\n", ITERATIONS);
    printf("Sequential: %lu us\n", seq_time);
    printf("Parallel: %lu us\n", par_time);
    printf("Results match: %s\n", ok ? "Yes" : "No");
    
    if (par_time > 0) {
        float speedup = (float)seq_time / par_time;
        printf("Speedup: %.2fx\n", speedup);
    } else {
        printf("Speedup: N/A (parallel time too small)\n");
    }
    
    printf("\nMatrix A:\n");
    for (int i = 0; i < 3; i++) {
        printf("%d %d %d\n", A[i*3], A[i*3+1], A[i*3+2]);
    }
    
    printf("\nMatrix B:\n");
    for (int i = 0; i < 3; i++) {
        printf("%d %d %d\n", B[i*3], B[i*3+1], B[i*3+2]);
    }
    
    printf("\nSequential Result:\n");
    for (int i = 0; i < 3; i++) {
        printf("%d %d %d\n", C_seq[i*3], C_seq[i*3+1], C_seq[i*3+2]);
    }
    
    printf("\nParallel Result:\n");
    for (int i = 0; i < 3; i++) {
        printf("%d %d %d\n", C_par[i*3], C_par[i*3+1], C_par[i*3+2]);
    }
    
    while (1) _waitx(_clockfreq());
    return 0;
}
