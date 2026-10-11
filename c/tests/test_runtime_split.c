#include "../runtime_split.h"
#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static size_t cases;
static uint64_t bits(double value) { uint64_t result; memcpy(&result,&value,sizeof result);return result; }
static double value(uint64_t raw) { double result;memcpy(&result,&raw,sizeof result);return result; }
static int finite_bits(double number) {return (bits(number)&UINT64_C(0x7ff0000000000000))!=UINT64_C(0x7ff0000000000000);}

/* Independent arithmetic oracle. Workload costs below are exact multiples of
 * 1/8, so this enumeration also stays independent of fast-math reassociation. */
static int reference(const uint32_t c[4],const double w[9],uint32_t n[2],double t[4]) {
    int found=0;
    for(uint32_t q=0;q<=c[2];++q) {
        uint32_t h=c[1]-q;
        if((h&&w[0]==0)||(q&&w[2]==0))continue;
        double host=h*w[0]+w[3],device=q*w[2]+w[4],serial=w[8];
        if(h||w[3]>0)host+=w[5];
        if(q||w[4]>0)device+=w[6];
        if(c[3])device+=q*w[1];else serial+=q*w[1];
        double total=serial+(host>device?host:device)+w[7];
        if(!finite_bits(host)||!finite_bits(device)||!finite_bits(serial)||!finite_bits(total))continue;
        if(!found||total<t[3]) {
            n[0]=q;n[1]=h;t[0]=host;t[1]=device;t[2]=serial;t[3]=total;found=1;
        }
    }
    return !found;
}

static void failure(const uint32_t *controls,const double *costs) {
    uint32_t counts[4]={UINT32_C(0x12345678),UINT32_C(0x76543210),11,17},saved_counts[4];
    double times[6],saved_times[6];
    memset(times,0xa5,sizeof times);memcpy(saved_times,times,sizeof times);memcpy(saved_counts,counts,sizeof counts);
    assert(coli_split_plan(controls,costs,counts+1,times+1)!=0);
    assert(!memcmp(counts,saved_counts,sizeof counts)&&!memcmp(times,saved_times,sizeof times));++cases;
}

static void compare(const uint32_t controls[4],const double costs[9]) {
    uint32_t expected_count[2]={0,0},actual_count[4]={37,0,0,41},control_before[4];
    double expected_time[4]={0,0,0,0},actual_time[6]={71,0,0,0,0,73},cost_before[9];
    memcpy(control_before,controls,sizeof control_before);memcpy(cost_before,costs,sizeof cost_before);
    if(reference(controls,costs,expected_count,expected_time)) {failure(controls,costs);return;}
    assert(coli_split_plan(controls,costs,actual_count+1,actual_time+1)==0);
    assert(!memcmp(actual_count+1,expected_count,sizeof expected_count));
    assert(!memcmp(actual_time+1,expected_time,sizeof expected_time));
    assert(actual_count[0]==37&&actual_count[3]==41&&actual_time[0]==71&&actual_time[5]==73);
    assert(!memcmp(controls,control_before,sizeof control_before)&&!memcmp(costs,cost_before,sizeof cost_before));++cases;
}

static void known_workloads(void) {
    uint32_t c[4]={1,12,12,1},n[2];double t[4],w[9]={10,1,2,0,0,0,0,0,0};
    assert(coli_split_plan(c,w,n,t)==0&&n[0]==9&&n[1]==3&&t[0]==30&&t[1]==27&&t[2]==0&&t[3]==30);++cases;
    c[3]=0;
    assert(coli_split_plan(c,w,n,t)==0&&n[0]==10&&n[1]==2&&t[0]==20&&t[1]==20&&t[2]==10&&t[3]==30);++cases;
    c[3]=1;w[6]=1000;
    assert(coli_split_plan(c,w,n,t)==0&&n[0]==0&&t[1]==0&&t[3]==120);++cases;
    w[4]=5;w[6]=100;
    assert(coli_split_plan(c,w,n,t)==0&&n[0]==1&&t[0]==110&&t[1]==108&&t[3]==110);++cases;
    c[2]=0;w[4]=0;w[5]=4;w[6]=1000;w[7]=6;w[8]=7;
    assert(coli_split_plan(c,w,n,t)==0&&n[0]==0&&n[1]==12&&t[0]==124&&t[1]==0&&t[2]==7&&t[3]==137);++cases;
    c[1]=c[2]=0;w[3]=5;w[4]=11;
    assert(coli_split_plan(c,w,n,t)==0&&t[0]==9&&t[1]==1011&&t[3]==1024);++cases;
    w[3]=w[4]=0;
    assert(coli_split_plan(c,w,n,t)==0&&t[0]==0&&t[1]==0&&t[3]==13);++cases;
    /* If fixed work already launches a device branch, its queue is charged
     * even with q=0; a missing per-expert profile still prohibits cold tasks. */
    w[0]=w[2]=0;w[3]=5;w[4]=11;compare(c,w);
    c[1]=3;c[2]=3;failure(c,w);
    w[2]=2;compare(c,w); /* only q=3 is viable */
    c[2]=2;failure(c,w);
    w[0]=2;w[2]=0;compare(c,w); /* only q=0 is viable */
    w[0]=value(UINT64_C(0x8000000000000000));failure(c,w);
    w[2]=2;c[2]=3;compare(c,w); /* negative zero is unavailable, not invalid */
}

static uint32_t random_state=UINT32_C(0x91f25eb7);
static uint32_t random32(void){random_state=random_state*UINT32_C(1664525)+UINT32_C(1013904223);return random_state;}
static void enumeration(void) {
    for(uint32_t m=0;m<=16;++m)for(uint32_t cap=0;cap<=m;++cap)for(uint32_t overlap=0;overlap<=1;++overlap)
        for(unsigned profile=0;profile<24;++profile) {
            uint32_t c[4]={COLI_SPLIT_ABI,m,cap,overlap};double w[9];
            for(unsigned i=0;i<9;++i)w[i]=(double)(random32()%65)/8;
            if(profile%3==0)w[3]=0;
            if(profile%4==0)w[4]=0;
            if(profile%7==0)w[0]=0;
            if(profile%11==0)w[2]=0;
            compare(c,w);
        }
    uint32_t c[4]={1,1024,1024,1};double w[9]={7,2,3,29,17,19,23,31,37};compare(c,w);
    c[3]=0;compare(c,w);c[2]=0;compare(c,w);
}

static void invalid_and_transactional(void) {
    uint32_t c[4]={1,7,5,1};double w[9]={3,2,1,4,5,6,7,8,9};
    const uint32_t bad[][4]={{0,7,5,1},{2,7,5,1},{1,1025,0,1},{1,7,8,1},{1,7,5,2},
                             {UINT32_MAX,0,0,0},{1,UINT32_MAX,0,0},{1,1,UINT32_MAX,0},{1,1,0,UINT32_MAX}};
    for(size_t k=0;k<sizeof bad/sizeof bad[0];++k)failure(bad[k],w);
    const uint64_t invalid[]={UINT64_C(0x7ff0000000000000),UINT64_C(0xfff0000000000000),
        UINT64_C(0x7ff8000000000000),UINT64_C(0x7ff0000000000001),UINT64_C(0xfff8000000000042),
        UINT64_C(0xbff0000000000000),UINT64_C(0x8000000000000001)};
    for(unsigned i=0;i<9;++i)for(size_t j=0;j<sizeof invalid/sizeof invalid[0];++j) {
        double saved=w[i];w[i]=value(invalid[j]);failure(c,w);w[i]=saved;
    }
    failure(NULL,w);failure(c,NULL);
    uint32_t n[2]={43,47};double t[4]={51,53,57,59},saved[4];memcpy(saved,t,sizeof t);
    assert(coli_split_plan(c,w,NULL,t)!=0&&!memcmp(t,saved,sizeof t));
    assert(coli_split_plan(c,w,n,NULL)!=0&&n[0]==43&&n[1]==47);cases+=2;
    union {double time[8];uint32_t count[16];} shared,before;memset(&shared,0x3c,sizeof shared);before=shared;
    assert(coli_split_plan(c,w,shared.count+2,shared.time+1)!=0&&!memcmp(&shared,&before,sizeof shared));++cases;
    /* Aliased input and output counts are safe because input is snapshotted. */
    uint32_t aliased[4]={1,3,3,1};double a[9]={10,1,4,0,0,0,0,0,0};
    assert(coli_split_plan(aliased,a,aliased+2,t)==0&&aliased[2]==2&&aliased[3]==1);++cases;
}

static void extremes_and_rounding(void) {
    uint32_t c[4]={1,2,2,1},n[2];double t[4],w[9]={DBL_MAX,0,DBL_MAX,0,0,0,0,0,0};
    assert(coli_split_plan(c,w,n,t)==0&&n[0]==1&&t[0]==DBL_MAX&&t[1]==DBL_MAX&&t[3]==DBL_MAX);++cases;
    w[7]=DBL_MAX;failure(c,w);w[7]=0;w[8]=DBL_MAX;failure(c,w);
    c[1]=1;c[2]=0;w[0]=10;w[2]=0;w[1]=DBL_MAX;w[6]=DBL_MAX;w[8]=0;compare(c,w);
    c[1]=0;w[0]=w[2]=0;w[3]=DBL_MAX;w[4]=0;w[5]=0;
    assert(coli_split_plan(c,w,n,t)==0&&t[0]==DBL_MAX&&t[3]==DBL_MAX);++cases;
    w[3]=0;w[6]=0;c[1]=3;w[0]=value(1);w[5]=value(2);
    assert(coli_split_plan(c,w,n,t)==0&&bits(t[0])==5&&bits(t[3])==5);++cases;
    c[1]=0;w[0]=0;w[3]=value(1);w[4]=value(1);w[5]=value(2);w[6]=value(1);w[7]=value(8);w[8]=value(4);
    assert(coli_split_plan(c,w,n,t)==0&&bits(t[0])==3&&bits(t[1])==2&&bits(t[3])==15);++cases;
    memset(w,0,sizeof w);c[1]=2;w[0]=value(UINT64_C(0x000fffffffffffff));
    assert(coli_split_plan(c,w,n,t)==0&&bits(t[0])==UINT64_C(0x001ffffffffffffe));++cases;
    c[1]=1;w[0]=1;w[3]=value(UINT64_C(0x3ca0000000000000));
    assert(coli_split_plan(c,w,n,t)==0&&bits(t[0])==UINT64_C(0x3ff0000000000000));++cases;
    w[0]=value(UINT64_C(0x3ff0000000000001));
    assert(coli_split_plan(c,w,n,t)==0&&bits(t[0])==UINT64_C(0x3ff0000000000002));++cases;
    w[3]=0;c[1]=3;
    assert(coli_split_plan(c,w,n,t)==0&&bits(t[0])==UINT64_C(0x4008000000000002));++cases;
    w[3]=value(1);c[1]=1;w[0]=1;
    assert(coli_split_plan(c,w,n,t)==0&&bits(t[0])==UINT64_C(0x3ff0000000000000));++cases;
}

#if !defined(__FAST_MATH__) && !defined(COLI_SPLIT_FAST_MATH)
/* Independent hardware binary64 checks across the complete exponent range,
 * including subnormals and rounding boundaries. Volatile separates stages. */
static void hardware_oracle(void) {
    for(unsigned k=0;k<12000;++k) {
        uint64_t aa=((uint64_t)random32()<<32)|random32(),bb=((uint64_t)random32()<<32)|random32();
        aa%=UINT64_C(0x7ff0000000000000);bb%=UINT64_C(0x7ff0000000000000);
        volatile double a=value(aa),b=value(bb),expected=a+b;
        uint32_t c[4]={1,1,0,1},n[2];double w[9]={a,0,0,b,0,0,0,0,0},t[4];
        if(!finite_bits(expected))failure(c,w);
        else {assert(coli_split_plan(c,w,n,t)==0&&bits(t[0])==bits(expected)&&bits(t[3])==bits(expected));++cases;}
        uint32_t count=random32()%1024+1;c[1]=count;w[3]=0;
        expected=a*(double)count;
        if(!finite_bits(expected))failure(c,w);
        else {assert(coli_split_plan(c,w,n,t)==0&&bits(t[0])==bits(expected)&&bits(t[3])==bits(expected));++cases;}
    }
}
#endif

int main(void) {
    known_workloads();enumeration();invalid_and_transactional();extremes_and_rounding();
#if !defined(__FAST_MATH__) && !defined(COLI_SPLIT_FAST_MATH)
    hardware_oracle();
#endif
    printf("native split policy: %zu independent enumeration, unavailable/queue, binary64 overflow/subnormal and transactional cases passed\n",cases);
    return 0;
}
