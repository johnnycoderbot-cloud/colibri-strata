/* Exact legacy parity, independent rows, tails/canaries, and optional warm
 * projection microbench. Compile standalone with strict warnings + OpenMP.
 * --bench-only reports kernel time, never model tokens/second. */
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#endif
#include <assert.h>
#include <time.h>
#include <stdio.h>
#include "../compat.h"
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "../idot.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

static uint32_t state=0x64c723a1u;
static uint32_t random_u32(void){state^=state<<13;state^=state>>17;state^=state<<5;return state;}
static int32_t mathematical_dot(const int8_t*w,const int8_t*x,int I){
    int64_t sum=0; for(int i=0;i<I;i++) sum+=(int32_t)w[i]*x[i]; return (int32_t)sum;
}
static void check_dot(int I){
    int stride=I+19;
    int8_t *weight=malloc((size_t)I+1),*activation=malloc((size_t)4*stride+3);
    assert(weight&&activation); int8_t*w=weight+1,*x=activation+3;
    for(int trial=0;trial<9;trial++){
        for(int i=0;i<I;i++) w[i]=(int8_t)random_u32();
        for(int i=0;i<4*stride;i++) x[i]=(int8_t)((int)(random_u32()%255)-127);
        if(trial==1) for(int i=0;i<I;i++) w[i]=-128;
        if(trial==2) for(int i=0;i<I;i++) w[i]=0;
        if(trial==3) for(int i=0;i<I;i++) w[i]=127;
        if(trial>=4) for(int r=0;r<4;r++) for(int i=0;i<I;i++)
            x[(int64_t)r*stride+i]=(trial==4||i%3==0)?-128:(int8_t)random_u32();
        for(int n=1;n<=4;n++){
            int32_t out[6]; for(int i=0;i<6;i++) out[i]=0x12345678;
            dot_i8i8_rows(w,x,I,stride,n,out+1);
            for(int r=0;r<n;r++){
                int32_t expected=dot_i8i8(w,x+(int64_t)r*stride,I);
                assert(out[r+1]==expected);
                if(trial<4) assert(expected==mathematical_dot(w,x+(int64_t)r*stride,I));
            }
            assert(out[0]==0x12345678);
            for(int i=n+1;i<6;i++) assert(out[i]==0x12345678);
        }
    }
    free(weight);free(activation);
}
static void check_driver(int S,int I,int O,int include_minimum){
    size_t count=(size_t)S*O, wcount=(size_t)I*O, xcount=(size_t)S*I;
    int8_t*q=malloc(wcount+1),*x=malloc(xcount+1);
    float *scale=malloc(((size_t)O+1)*sizeof(float)),*sx=malloc(((size_t)S+1)*sizeof(float));
    float *storage=malloc((count+4)*sizeof(float)),*legacy=malloc((count+4)*sizeof(float));
    assert(q&&x&&scale&&sx&&storage&&legacy);
    for(size_t i=0;i<wcount;i++) q[i]=(int8_t)random_u32();
    for(size_t i=0;i<xcount;i++) x[i]=include_minimum?(int8_t)random_u32():(int8_t)((int)(random_u32()%255)-127);
    if(include_minimum&&wcount&&xcount){q[0]=-128;x[0]=-128;}
    for(int o=0;o<O;o++) scale[o]=(float)(1+random_u32()%1000)*.0000031f;
    for(int s=0;s<S;s++) sx[s]=(float)(1+random_u32()%1000)*.0000073f;
    for(size_t i=0;i<count+4;i++) storage[i]=legacy[i]=-9876.5f;
    assert(!setenv("COLI_IDOT_ROWS","0",1));
    matmul_q_idot(legacy+2,x,sx,q,scale,S,I,O);
    assert(!setenv("COLI_IDOT_ROWS","1",1));
    matmul_q_idot(storage+2,x,sx,q,scale,S,I,O);
    assert(!memcmp(storage,legacy,(count+4)*sizeof(float)));
    for(int s=0;s<S;s++) for(int o=0;o<O;o++){
        int32_t integer=dot_i8i8(q+(int64_t)o*I,x+(int64_t)s*I,I);
        float expected=(float)integer*scale[o]*sx[s];
        assert(!memcmp(&expected,storage+2+(int64_t)s*O+o,sizeof(float)));
    }
    if(S>=2&&I&&O){
        /* Changing one activation row must not change any other row. */
        int changed=S/2; x[(int64_t)changed*I]^=0x5f;
        matmul_q_idot(storage+2,x,sx,q,scale,S,I,O);
        for(int s=0;s<S;s++) if(s!=changed)
            assert(!memcmp(storage+2+(int64_t)s*O,legacy+2+(int64_t)s*O,(size_t)O*sizeof(float)));
    }
    assert(storage[0]==-9876.5f&&storage[1]==-9876.5f&&storage[count+2]==-9876.5f&&storage[count+3]==-9876.5f);
    free(q);free(x);free(scale);free(sx);free(storage);free(legacy);
}

static double seconds(void){struct timespec ts;assert(!clock_gettime(CLOCK_MONOTONIC,&ts));return ts.tv_sec+ts.tv_nsec*1e-9;}
static int compare_double(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static volatile float benchmark_sink;
static void benchmark(int S,int I,int O){
    size_t count=(size_t)S*O;
    int8_t*q=malloc((size_t)I*O),*x=malloc((size_t)S*I);
    float*y=malloc(count*sizeof(float)),*sx=malloc((size_t)S*sizeof(float)),*scale=malloc((size_t)O*sizeof(float));
    assert(q&&x&&y&&sx&&scale);
    for(size_t i=0;i<(size_t)I*O;i++) q[i]=(int8_t)random_u32();
    for(size_t i=0;i<(size_t)S*I;i++) x[i]=(int8_t)((int)(random_u32()%255)-127);
    for(int s=0;s<S;s++) sx[s]=.0023f;
    for(int o=0;o<O;o++) scale[o]=.0017f;
    double times[2][9];
    for(int round=0;round<9;round++) for(int phase=0;phase<2;phase++){
        int mode=(phase+round)%2; assert(!setenv("COLI_IDOT_ROWS",mode?"1":"0",1));
        for(int warm=0;warm<3;warm++) matmul_q_idot(y,x,sx,q,scale,S,I,O);
        double start=seconds();
        for(int repeat=0;repeat<16;repeat++){
#ifdef __GNUC__
            __asm__ volatile("" ::: "memory");
#endif
            matmul_q_idot(y,x,sx,q,scale,S,I,O);
            benchmark_sink+=y[(size_t)repeat%count];
        }
        times[mode][round]=(seconds()-start)/16.;
    }
    qsort(times[0],9,sizeof(double),compare_double);qsort(times[1],9,sizeof(double),compare_double);
    int threads=1;
#ifdef _OPENMP
    threads=omp_get_max_threads();
#endif
    printf("{\"kernel\":\"%s\",\"threads\":%d,\"S\":%d,\"I\":%d,\"O\":%d,\"legacy_seconds\":%.9g,\"tiled_seconds\":%.9g,\"ratio\":%.6g}\n",IDOT_KERNEL,threads,S,I,O,times[0][4],times[1][4],times[0][4]/times[1][4]);
    free(q);free(x);free(y);free(sx);free(scale);
}
int main(int argc,char**argv){
    if(argc>1&&!strcmp(argv[1],"--bench-only")){
        static const int inputs[]={1024,3584,6144},rows[]={1,2,3,4,9};
        for(size_t i=0;i<sizeof inputs/sizeof inputs[0];i++)
            for(size_t s=0;s<sizeof rows/sizeof rows[0];s++) benchmark(rows[s],inputs[i],4096);
        return 0;
    }
    static const int inputs[]={0,1,2,15,16,17,31,32,33,63,64,65,127,128,129,257,1024,3584,6144,6145};
    static const int outputs[]={1,2,3,7,17},rows[]={1,2,3,4,5,9,17,32};
    for(size_t i=0;i<sizeof inputs/sizeof inputs[0];i++) check_dot(inputs[i]);
    for(size_t i=0;i<sizeof inputs/sizeof inputs[0];i++)
        for(size_t o=0;o<sizeof outputs/sizeof outputs[0];o++)
            for(size_t s=0;s<sizeof rows/sizeof rows[0];s++)
                for(int minimum=0;minimum<2;minimum++) check_driver(rows[s],inputs[i],outputs[o],minimum);
    /* Dense projection geometries and large odd output tails. */
    check_driver(2,1024,3584,0);check_driver(3,1024,3585,1);
    check_driver(3,3584,1024,0);check_driver(4,3584,1025,1);
    check_driver(2,6144,1025,0);check_driver(5,33,6145,1);
    assert(!unsetenv("COLI_IDOT_ROWS"));
    printf("idot multi-row exact legacy parity (%s): ok\n",IDOT_KERNEL);
    return 0;
}
