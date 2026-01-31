#include <stdio.h>
#include <propeller2.h>
#include <stdint.h>

#define STACKSIZE 256

unsigned char stack1[STACKSIZE];
unsigned char stack2[STACKSIZE];
unsigned char stack3[STACKSIZE];
unsigned char stack4[STACKSIZE];
unsigned char stack5[STACKSIZE];
unsigned char stack6[STACKSIZE];
unsigned char stack7[STACKSIZE];

typedef struct {
    volatile int num1;
    volatile int num2;
    volatile int ready1;
    volatile int ready2;
    volatile int result;
    volatile int readyRes;
} SharedPair;

SharedPair groupA;
SharedPair groupB;

typedef struct {
    SharedPair *shared;
    unsigned seed_const;
} CogParam;

static inline uint32_t xorshift32(uint32_t seed) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
}

void cog_random_func(void *p)
{
    CogParam *par = (CogParam *)p;
    SharedPair *s = par->shared;

    uint32_t seed = (uint32_t)(_cnt()) ^ par->seed_const;
    if (seed == 0) seed = 0xCAFEBABE;

    volatile int *numPtr;
    volatile int *readyPtr;

    if ((par->seed_const & 1) == 1)
        numPtr = &s->num1, readyPtr = &s->ready1;
    else
        numPtr = &s->num2, readyPtr = &s->ready2;

    for (;;) {
        if (!(*readyPtr)) {
            seed = xorshift32(seed ^ _cnt());
            *numPtr = (int)((seed % 20u) + 1u);
            *readyPtr = 1;
        }
        _waitx(_clockfreq() / 5);
    }
}

void cog_multiplier_func(void *p)
{
    SharedPair *s = (SharedPair *)p;

    for (;;) {
        if (s->readyRes) {
            _waitx(_clockfreq() / 50);
            continue;
        }

        if (s->ready1 && s->ready2) {
            int a = s->num1;
            int b = s->num2;

            if (a < 1 || a > 20 || b < 1 || b > 20) {
                s->ready1 = 0;
                s->ready2 = 0;
                continue;
            }

            int r = a * b;
            s->result = r;
            s->readyRes = 1;
            printf("Cog mult: %d x %d = %d\n", a, b, r);

            s->ready1 = 0;
            s->ready2 = 0;

            _waitx(_clockfreq() / 5);
        } else {
            _waitx(_clockfreq() / 20);
        }
    }
}

void cog_final_func(void *p)
{
    struct {
        SharedPair *A;
        SharedPair *B;
        int *cogIDs;
    } *par = p;

    SharedPair *a = par->A;
    SharedPair *b = par->B;
    int *cogIDs = par->cogIDs;

    int count = 0;

    for (;;) {
        if (a->readyRes && b->readyRes) {
            int r1 = a->result;
            int r2 = b->result;
            int final = r1 * r2;
            count++;

            printf("Cog 7 FINAL %d: (%d) x (%d) = %d\n", count, r1, r2, final);

            a->readyRes = 0;
            b->readyRes = 0;

            if (count >= 4) {
                printf("\nCog 7: reached 4 cycle limit. Stopping all COGs...\n");

                for (int i = 0; i < 7; i++) {
                    if (cogIDs[i] != _cogid() && cogIDs[i] >= 0)
                        _cogstop(cogIDs[i]);
                }

                _waitx(_clockfreq() / 4);
                printf("All COGs stopped.\n");
                _cogstop(_cogid());
            }

            _waitx(_clockfreq() / 2);
        } else {
            _waitx(_clockfreq() / 10);
        }
    }
}

void main()
{
    groupA.num1 = groupA.num2 = groupA.ready1 = groupA.ready2 = groupA.readyRes = 0;
    groupB.num1 = groupB.num2 = groupB.ready1 = groupB.ready2 = groupB.readyRes = 0;

    CogParam p1 = { &groupA, 0x1111 };
    CogParam p2 = { &groupA, 0x2222 };
    CogParam p4 = { &groupB, 0x4444 };
    CogParam p5 = { &groupB, 0x5555 };

    int cogs[7];
    cogs[0] = _cogstart_C(cog_random_func, &p1, stack1, sizeof(stack1));
    cogs[1] = _cogstart_C(cog_random_func, &p2, stack2, sizeof(stack2));
    cogs[2] = _cogstart_C(cog_multiplier_func, &groupA, stack3, sizeof(stack3));
    cogs[3] = _cogstart_C(cog_random_func, &p4, stack4, sizeof(stack4));
    cogs[4] = _cogstart_C(cog_random_func, &p5, stack5, sizeof(stack5));
    cogs[5] = _cogstart_C(cog_multiplier_func, &groupB, stack6, sizeof(stack6));

    struct { SharedPair *A; SharedPair *B; int *cogIDs; } finalParam = { &groupA, &groupB, cogs };
    cogs[6] = _cogstart_C(cog_final_func, &finalParam, stack7, sizeof(stack7));

    printf("Started cogs: %d,%d,%d,%d,%d,%d,%d\n", 
            cogs[0], cogs[1], cogs[2], cogs[3], cogs[4], cogs[5], cogs[6]);

    for (;;)
        _waitx(_clockfreq());
}
