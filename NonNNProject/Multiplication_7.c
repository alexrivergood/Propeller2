#include <stdio.h>
#include <propeller2.h>
#include <stdint.h>

#define STACKSIZE 256

unsigned char stack3[STACKSIZE];
unsigned char stack6[STACKSIZE];
unsigned char stack7[STACKSIZE];

typedef struct {
    volatile int num1;
    volatile int num2;
    volatile int result;
    volatile int done;
} SharedPair;

SharedPair groupA;
SharedPair groupB;

typedef struct {
    SharedPair *A;
    SharedPair *B;
} FinalParam;

void cog_multiplierA(void *p)
{
    SharedPair *s = (SharedPair *)p;
    s->result = s->num1 * s->num2;
    printf("\nCOG3: %d x %d = %d\n", s->num1, s->num2, s->result);
    s->done = 1;
    _cogstop(_cogid());
}

void cog_multiplierB(void *p)
{
    SharedPair *s = (SharedPair *)p;
    s->result = s->num1 * s->num2;
    printf("\nCOG6: %d x %d = %d\n", s->num1, s->num2, s->result);
    s->done = 1;
    _cogstop(_cogid());
}

void cog_final(void *p)
{
    FinalParam *fp = (FinalParam *)p;
    SharedPair *a = fp->A;
    SharedPair *b = fp->B;

    while (!(a->done && b->done)) {
        _waitx(_clockfreq() / 10);
    }

    int final = a->result * b->result;
    printf("\nCOG7 FINAL: (%d) x (%d) = %d\n", a->result, b->result, final);

    printf("Done");
    _cogstop(_cogid());
}

void main()
{
    groupA.num1 = groupA.num2 = groupA.result = groupA.done = 0;
    groupB.num1 = groupB.num2 = groupB.result = groupB.done = 0;

    printf("Enter number for COG1: ");
    scanf("%d", (int *)&groupA.num1);
    printf("Enter number for COG2: ");
    scanf("%d", (int *)&groupA.num2);

    printf("Enter number for COG4: ");
    scanf("%d", (int *)&groupB.num1);
    printf("Enter number for COG5: ");
    scanf("%d", (int *)&groupB.num2);

    int c3 = _cogstart_C(cog_multiplierA, &groupA, stack3, sizeof(stack3));
    int c6 = _cogstart_C(cog_multiplierB, &groupB, stack6, sizeof(stack6));

    while (!(groupA.done && groupB.done)) {
        _waitx(_clockfreq() / 10);
    }

    FinalParam fp = { &groupA, &groupB };
    int c7 = _cogstart_C(cog_final, &fp, stack7, sizeof(stack7));

    printf("\nCogs started: %d, %d, %d\n", c3, c6, c7);

    for (;;)
        _waitx(_clockfreq());
}
