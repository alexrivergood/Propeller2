//The weights here are quantized
//Run time:30s
//Weight: 400KB
//No use of parallelism
//4 cogs are used, one for main, another for conv1 + pool, another for conv2 + pool + fc2, another for fc1 + softmax
//RELU and softmax have been used

#include <propeller2.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "mnist_cnn_weights_q.h"
#include "numero_matriz.h"

unsigned char stkA[256], stkB[512], stkC[256];

#define IN_H 28
#define IN_W 28
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
#define FC_IN (64*5*5)
#define FC1_OUT 128
#define FC2_OUT 10

typedef struct {
    float img2d[IN_H][IN_W];
    float c1p[C1_OUT*C1_P_H*C1_P_W];
    float c2p[C2_OUT*C2_P_H*C2_P_W];
    float fc1[FC1_OUT];
    float out[FC2_OUT];
    volatile int ready_input;
    volatile int ready_mid;
    volatile int ready_out;
    volatile int seq_input;
    volatile int seq_mid;
    volatile int seq_out;
    volatile int done_seq;
} NN;

NN nn;

static inline float relu(float x){return x>0?x:0;}

void softmax(float*x,int n){
    float m=x[0];
    for(int i=1;i<n;i++) if(x[i]>m)m=x[i];
    float s=0;
    for(int i=0;i<n;i++){ x[i]=expf(x[i]-m); s+=x[i]; }
    for(int i=0;i<n;i++) x[i]/=s;
}

static inline int i1(int o,int i){return o*FC_IN+i;}
static inline int i2(int o,int i){return o*FC1_OUT+i;}
static inline int id1(int o,int i,int y,int x){return ((o*1+i)*3+y)*3+x;}
static inline int id2(int o,int i,int y,int x){return ((o*32+i)*3+y)*3+x;}

void conv(const float*in,int ic,int ih,int iw,float*out,int oc,int k,
          const int8_t*w,float ws,const float*b,
          int(*id)(int,int,int,int)){
    int oh=ih-(k-1), ow=iw-(k-1);
    for(int o=0;o<oc;o++)
    for(int y=0;y<oh;y++)
    for(int x=0;x<ow;x++){
        float s=b[o];
        for(int i=0;i<ic;i++)
        for(int ky=0;ky<k;ky++)
        for(int kx=0;kx<k;kx++)
            s+=in[(i*ih+y+ky)*iw+(x+kx)]*((float)w[id(o,i,ky,kx)]*ws);
        out[(o*oh+y)*ow+x]=s;
    }
}

void pool(const float*in,int c,int ih,int iw,float*out){
    int oh=ih>>1, ow=iw>>1;
    for(int o=0;o<c;o++)
    for(int y=0;y<oh;y++)
    for(int x=0;x<ow;x++){
        int yy=y*2, xx=x*2;
        float m=in[(o*ih+yy)*iw+xx];
        float v;
        v=in[(o*ih+yy)*iw+xx+1]; if(v>m)m=v;
        v=in[(o*ih+yy+1)*iw+xx]; if(v>m)m=v;
        v=in[(o*ih+yy+1)*iw+xx+1]; if(v>m)m=v;
        out[(o*oh+y)*ow+x]=relu(m);
    }
}

void cog_input(void*p){
    nn.ready_input=1;
    int last=0;
    while(1){
        if(nn.seq_input!=last){
            last=nn.seq_input;
            for(int y=0;y<IN_H;y++)
            for(int x=0;x<IN_W;x++)
                nn.img2d[y][x]=numero[y][x];
            nn.seq_mid=last;
        }
    }
}

void cog_mid(void*p){
    static float c1[C1_OUT*C1_H*C1_W];
    static float c2[C2_OUT*C2_H*C2_W];
    nn.ready_mid=1;
    int last=0;
    while(1){
        if(nn.seq_mid!=last){
            last=nn.seq_mid;
            conv((float*)nn.img2d,1,28,28,c1,32,3,conv1_weight_q,conv1_weight_scale,conv1_bias,id1);
            pool(c1,32,26,26,nn.c1p);
            conv(nn.c1p,32,13,13,c2,64,3,conv2_weight_q,conv2_weight_scale,conv2_bias,id2);
            pool(c2,64,11,11,nn.c2p);
            for(int o=0;o<FC1_OUT;o++){
                float s=fc1_bias[o];
                for(int i=0;i<FC_IN;i++)
                    s+=nn.c2p[i]*((float)fc1_weight_q[i1(o,i)]*fc1_weight_scale);
                nn.fc1[o]=relu(s);
            }
            nn.seq_out=last;
        }
    }
}

void cog_out(void*p){
    nn.ready_out=1;
    int last=0;
    while(1){
        if(nn.seq_out!=last){
            last=nn.seq_out;
            for(int o=0;o<FC2_OUT;o++){
                float s=fc2_bias[o];
                for(int i=0;i<FC1_OUT;i++)
                    s+=nn.fc1[i]*((float)fc2_weight_q[i2(o,i)]*fc2_weight_scale);
                nn.out[o]=s;
            }
            softmax(nn.out,FC2_OUT);
            nn.done_seq=last;
        }
    }
}

int main(){
    printf("\n Resultados secuenciales\n");

    nn.ready_input=0; nn.ready_mid=0; nn.ready_out=0;
    nn.seq_input=0; nn.seq_mid=0; nn.seq_out=0; nn.done_seq=0;

    _cogstart_C(cog_input,&nn,stkA,256);
    _cogstart_C(cog_mid,&nn,stkB,512);
    _cogstart_C(cog_out,&nn,stkC,256);

    while(!nn.ready_input || !nn.ready_mid || !nn.ready_out);

    uint64_t t0=_getms();
    nn.seq_input++;
    while(nn.done_seq!=nn.seq_input);
    uint64_t t1=_getms();

    int pm=0;
    for(int i=1;i<10;i++) if(nn.out[i]>nn.out[pm]) pm=i;

    printf("Test1 prob:\n");
    for(int i=0;i<10;i++)
        printf("%d: %.6f\n", i, nn.out[i]);
    printf("Prediccion: %d\n", pm);
    printf("Tiempo: %.3f ms\n", (double)(t1 - t0));

    uint64_t t2=_getms();
    nn.seq_input++;
    while(nn.done_seq!=nn.seq_input);
    uint64_t t3=_getms();

    pm=0;
    for(int i=1;i<10;i++) if(nn.out[i]>nn.out[pm]) pm=i;

    printf("Test2 prob:\n");
    for(int i=0;i<10;i++)
        printf("%d: %.6f\n", i, nn.out[i]);
    printf("Prediccion: %d\n", pm);
    printf("Tiempo: %.3f ms\n", (double)(t3 - t2));

    printf("OK\n");

    while(1);
}
