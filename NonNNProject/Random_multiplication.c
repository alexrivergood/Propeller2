#include <stdio.h>
#include <propeller2.h>
#include <stdint.h>

#define STACKSIZE 256

unsigned char stack1[STACKSIZE];
unsigned char stack2[STACKSIZE];
unsigned char stack3[STACKSIZE];

typedef struct {
    volatile int num1;
    volatile int num2;
    volatile int ready1;
    volatile int ready2;
} SharedData;

SharedData shared;

typedef struct {
    SharedData *shared;
    unsigned seed_const;
} CogParam;

CogParam param1, param2;

void cog1_func(void *p)
{
    CogParam *par = (CogParam *)p;
    SharedData *s = par->shared;

    uint32_t seed = (uint32_t)(_cnt()) ^ par->seed_const;
    if (seed == 0) seed = 0xDEADBEEF;

    for (;;) {
        if (!s->ready1) {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;

            s->num1 = (int)((seed % 20u) + 1u);
            s->ready1 = 1;
        }
        _waitx(_clockfreq() / 5);
    }
}

void cog2_func(void *p)
{
    CogParam *par = (CogParam *)p;
    SharedData *s = par->shared;

    uint32_t seed = (uint32_t)(_cnt()) ^ par->seed_const;
    if (seed == 0) seed = 0xC0FFEE;

    for (;;) {
        if (!s->ready2) {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;

            s->num2 = (int)((seed % 20u) + 1u);
            s->ready2 = 1;
        }
        _waitx(_clockfreq() / 5);
    }
}

void cog3_func(void *p)
{
    SharedData *s = (SharedData *)p;

    for (;;) {
        if (s->ready1 && s->ready2) {
            int a = s->num1;
            int b = s->num2;
            int result = a * b;
            printf("Cog 3: %d x %d = %d\n", a, b, result);

            s->ready1 = 0;
            s->ready2 = 0;

            _waitx(_clockfreq());
        } else {
            _waitx(_clockfreq() / 10);
        }
    }
}

void main()
{
    shared.num1 = 0;
    shared.num2 = 0;
    shared.ready1 = 0;
    shared.ready2 = 0;

    param1.shared = &shared;
    param1.seed_const = 0x1111;
    param2.shared = &shared;
    param2.seed_const = 0x2222;

    int c1 = _cogstart_C(cog1_func, &param1, stack1, sizeof(stack1));
    int c2 = _cogstart_C(cog2_func, &param2, stack2, sizeof(stack2));
    int c3 = _cogstart_C(cog3_func, &shared, stack3, sizeof(stack3));

    printf("Started cogs %d, %d, %d\n", c1, c2, c3);

    for (;;)
        ;
}
