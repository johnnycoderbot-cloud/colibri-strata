/* Optional wide S1 integer-dot gate and large-matrix microbenchmark.
 * Default: exactness/dispatch tests only. --bench: paired CPU matrix timings,
 * NOT a model/token-speed claim. Compile -march=native on the target host. */
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "../compat.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static unsigned wide_switch_reads;
static char *counted_getenv(const char *name){
    if(!strcmp(name,"COLI_IDOT_WIDE")) wide_switch_reads++;
    return getenv(name);
}
#define getenv counted_getenv
#include "../idot.h"
#undef getenv

#if defined(__AVXVNNI__) && defined(__AVX2__) && !defined(__AVX512VNNI__)
#define WIDE_QUALIFIED 1
#else
#define WIDE_QUALIFIED 0
#endif

static uint32_t random_state=0x904ea617u;
static uint32_t rnd(void){
    random_state^=random_state<<13;random_state^=random_state>>17;
    random_state^=random_state<<5;return random_state;
}
static void switch_set(const char *value){
#ifdef _WIN32
    assert(!_putenv_s("COLI_IDOT_WIDE",value?value:""));
#else
    if(value)assert(!setenv("COLI_IDOT_WIDE",value,1));
    else assert(!unsetenv("COLI_IDOT_WIDE"));
#endif
}

static void dot_gates(void){
    const int sizes[]={0,1,7,15,16,17,31,32,33,63,64,65,127,128,129,
                       255,256,257,1023,1024,1025,3583,3584,3585,6143,6144,6145};
    for(size_t k=0;k<sizeof sizes/sizeof sizes[0];k++){
        int I=sizes[k];
        int8_t *wb=malloc((size_t)I+9),*xb=malloc((size_t)I+9);assert(wb&&xb);
        /* Unaligned addresses exercise all vector boundaries without padding
         * the activation/weight stride to an assumed SIMD width. */
        int8_t *w=wb+3,*x=xb+5;
        for(int pattern=0;pattern<6;pattern++){
            memset(wb,0x37,(size_t)I+9);memset(xb,0x49,(size_t)I+9);
            for(int i=0;i<I;i++){
                w[i]=(int8_t)((int)(rnd()%256)-128);
                x[i]=(int8_t)((int)(rnd()%255)-127);
                if(pattern==1)w[i]=-128;
                if(pattern==2){w[i]=(int8_t)(i&1?-128:127);x[i]=(int8_t)(i&2?-127:127);}
                if(pattern==3)x[i]=-128; /* legacy equality, outside qrow contract */
                if(pattern==4){w[i]=-128;x[i]=-128;}
                if(pattern==5){w[i]=0;x[i]=0;}
            }
            int32_t old=dot_i8i8(w,x,I),wide=dot_i8i8_wide(w,x,I);
            assert(old==wide);
            if(pattern<3||pattern==5){
                int64_t exact=0;for(int i=0;i<I;i++)exact+=(int32_t)w[i]*x[i];
                assert(wide==(int32_t)exact);
            }
            assert(wb[0]==0x37&&wb[2]==0x37&&wb[I+3]==0x37);
            assert(xb[0]==0x49&&xb[4]==0x49&&xb[I+5]==0x49);
        }
        free(wb);free(xb);
    }
}

static void matrix_gate(int S,int I,int O){
    size_t values=(size_t)S*I,weights=(size_t)O*I,cells=(size_t)S*O;
    int8_t *x=malloc(values),*q=malloc(weights);float *sx=malloc((size_t)S*sizeof(float));
    float *sc=malloc((size_t)O*sizeof(float)),*want=malloc(cells*sizeof(float));
    float *yb=malloc((cells+2)*sizeof(float));assert(x&&q&&sx&&sc&&want&&yb);
    float *y=yb+1;
    for(size_t i=0;i<values;i++)x[i]=(int8_t)((int)(rnd()%256)-128);
    for(size_t i=0;i<weights;i++)q[i]=(int8_t)((int)(rnd()%256)-128);
    for(int s=0;s<S;s++)sx[s]=(float)(s+1)/63.f;
    for(int o=0;o<O;o++)sc[o]=(float)(o%17+1)/91.f;
    for(int s=0;s<S;s++)for(int o=0;o<O;o++)
        want[(size_t)s*O+o]=(float)dot_i8i8(q+(size_t)o*I,x+(size_t)s*I,I)*sc[o]*sx[s];
    for(int enabled=0;enabled<2;enabled++){
        switch_set(enabled?"1":"0");wide_switch_reads=0;
        yb[0]=-731.f;yb[cells+1]=913.f;
        matmul_q_idot(y,x,sx,q,sc,S,I,O);
        assert(!memcmp(want,y,cells*sizeof(float)));
        assert(wide_switch_reads==(unsigned)(WIDE_QUALIFIED&&S==1));
        assert(yb[0]==-731.f&&yb[cells+1]==913.f);
        float *second=malloc(cells*sizeof(float));assert(second);
        IdotProjection p[]={{y,q,sc,O},{second,q,sc,O}};
        wide_switch_reads=0;assert(matmul_q_idot_group(p,2,x,sx,S,I));
        assert(wide_switch_reads==(unsigned)(WIDE_QUALIFIED&&S==1));
        assert(!memcmp(want,y,cells*sizeof(float))&&!memcmp(want,second,cells*sizeof(float)));
        assert(yb[0]==-731.f&&yb[cells+1]==913.f);free(second);
    }
    free(x);free(q);free(sx);free(sc);free(want);free(yb);
}

static void switch_gates(void){
    const char *values[]={NULL,"0","1","garbage"};
    for(size_t i=0;i<sizeof values/sizeof values[0];i++){
        switch_set(values[i]);wide_switch_reads=0;
        assert(idot_wide_on()==(WIDE_QUALIFIED&&i==2));
        assert(wide_switch_reads==(unsigned)WIDE_QUALIFIED);
    }
    switch_set(NULL);
}

#ifdef _OPENMP
static double median(double *values,int n){
    for(int i=1;i<n;i++){double v=values[i];int j=i;while(j&&values[j-1]>v){values[j]=values[j-1];j--;}values[j]=v;}
    return values[n/2];
}
static double timed_matrix(float *y,const int8_t *x,const float *sx,const int8_t *q,
                           const float *sc,int I,int O,int enabled){
    switch_set(enabled?"1":"0");
    double start=omp_get_wtime();matmul_q_idot(y,x,sx,q,sc,1,I,O);
    double elapsed=omp_get_wtime()-start;assert(isfinite(elapsed)&&elapsed>0);return elapsed;
}
static void bench_shape(const char *label,int I,int O,int threads){
    enum {N=9};
    int8_t *x=malloc((size_t)I),*q=malloc((size_t)I*O);
    float *sc=malloc((size_t)O*sizeof(float)),*a=malloc((size_t)O*sizeof(float)),*b=malloc((size_t)O*sizeof(float));
    assert(x&&q&&sc&&a&&b);float sx=1.f/127.f;
    for(int i=0;i<I;i++)x[i]=(int8_t)((int)(rnd()%255)-127);
    for(size_t i=0;i<(size_t)I*O;i++)q[i]=(int8_t)((int)(rnd()%256)-128);
    for(int o=0;o<O;o++)sc[o]=(float)(o%13+1)/1024.f;
    omp_set_dynamic(0);omp_set_num_threads(threads);int actual=0;
    #pragma omp parallel
    {
        #pragma omp single
        actual=omp_get_num_threads();
    }
    assert(actual==threads);
    timed_matrix(a,x,&sx,q,sc,I,O,0);timed_matrix(b,x,&sx,q,sc,I,O,1);
    assert(!memcmp(a,b,(size_t)O*sizeof(float)));
    for(int warm=0;warm<3;warm++){
        timed_matrix(a,x,&sx,q,sc,I,O,0);timed_matrix(b,x,&sx,q,sc,I,O,1);
    }
    double old[N],wide[N],ratios[N],sorted_old[N],sorted_wide[N];
    for(int n=0;n<N;n++){
        if(n&1){wide[n]=timed_matrix(b,x,&sx,q,sc,I,O,1);old[n]=timed_matrix(a,x,&sx,q,sc,I,O,0);}
        else{old[n]=timed_matrix(a,x,&sx,q,sc,I,O,0);wide[n]=timed_matrix(b,x,&sx,q,sc,I,O,1);}
        assert(!memcmp(a,b,(size_t)O*sizeof(float)));ratios[n]=old[n]/wide[n];
        sorted_old[n]=old[n];sorted_wide[n]=wide[n];
    }
    double old_median=median(sorted_old,N),wide_median=median(sorted_wide,N);
    double paired_median=median(ratios,N);
    printf("{\"kind\":\"cpu_matrix_microbenchmark\",\"label\":\"%s\",\"I\":%d,\"O\":%d,\"S\":1,\"threads\":%d,\"weight_bytes\":%zu,\"exact\":true,\"old_ms\":%.6f,\"wide_ms\":%.6f,\"median_paired_ratio\":%.6f,\"pairs\":[",label,I,O,actual,(size_t)I*O,old_median*1000,wide_median*1000,paired_median);
    for(int n=0;n<N;n++)printf("%s[%.6f,%.6f]",n?",":"",old[n]*1000,wide[n]*1000);
    puts("]}");fflush(stdout);free(x);free(q);free(sc);free(a);free(b);
}
#endif

int main(int argc,char **argv){
    if(argc==2&&!strcmp(argv[1],"--bench")){
#if WIDE_QUALIFIED && defined(_OPENMP)
        for(int threads=1;threads<=8;threads+=7){
            bench_shape("lm_head",1024,248320,threads);
            bench_shape("dense_ffn_gate_up",1024,3584,threads);
        }
        switch_set(NULL);return 0;
#else
        fputs("wide benchmark requires AVX-VNNI (without AVX512VNNI) and OpenMP\n",stderr);return 2;
#endif
    }
    if(argc!=1){fputs("usage: test_idot_wide [--bench]\n",stderr);return 2;}
#ifdef _OPENMP
    omp_set_dynamic(0);omp_set_num_threads(2);
#endif
    switch_gates();dot_gates();
    const int inputs[]={1,7,15,16,17,31,32,33,63,64,65,127,128,129,1024,3584,6144};
    const int rows[]={1,2,3,5};
    for(size_t i=0;i<sizeof inputs/sizeof inputs[0];i++)
        for(size_t s=0;s<sizeof rows/sizeof rows[0];s++)matrix_gate(rows[s],inputs[i],33);
    switch_set(NULL);puts("idot wide: legacy dots, S1/group scale order, tails, canaries, per-dispatch switch and S2+ unchanged ok");
    return 0;
}
