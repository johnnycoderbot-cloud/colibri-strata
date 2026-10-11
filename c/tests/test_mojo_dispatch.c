/* Native Mojo/C ABI equivalence and deliberate incompatible-library fixtures.
 * Fixture libraries validate loader refusals; only the real .mojo shared
 * library exercises planner execution. No device-kernel/speed qualification. */
#define _POSIX_C_SOURCE 200809L
#include "../mojo_split_loader.h"
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(COLI_MOJO_TEST_WRONG_ABI) || defined(COLI_MOJO_TEST_MISSING_SYMBOLS) || defined(COLI_MOJO_TEST_BAD_INIT)
uint32_t coli_mojo_split_abi(void) {
#ifdef COLI_MOJO_TEST_WRONG_ABI
    return 99;
#else
    return COLI_SPLIT_ABI;
#endif
}
#ifndef COLI_MOJO_TEST_MISSING_SYMBOLS
int32_t coli_mojo_split_init(void) {
#ifdef COLI_MOJO_TEST_BAD_INIT
    return -1;
#else
    return 0;
#endif
}
int coli_mojo_split_plan(const uint32_t *a,const double *b,uint32_t *c,double *d) {
    (void)a;(void)b;(void)c;(void)d; return 0;
}
#endif
#else
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

static unsigned checks;
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr,"check %u failed at %s:%d: %s\n",checks,__FILE__,__LINE__,#x); exit(1); } } while (0)
#ifndef _WIN32
static unsigned comparisons;
static uint64_t random_state = UINT64_C(0x52cb90c89911da73);
static uint64_t random_bits(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 7;
    random_state ^= random_state << 17;
    return random_state;
}
static double from_bits(uint64_t value) {
    double result; memcpy(&result,&value,sizeof result); return result;
}
static void compare(ColiSplitPlanner mojo,const uint32_t *control,const double *cost) {
    uint32_t cc[2]={17,19},mc[2]={17,19};
    double ct[4]={23,29,31,37},mt[4]={23,29,31,37};
    int cr=coli_split_plan(control,cost,cc,ct),mr=mojo(control,cost,mc,mt);
    ++comparisons;
    CHECK(cr==mr);
    CHECK(memcmp(cc,mc,sizeof cc)==0);
    CHECK(memcmp(ct,mt,sizeof ct)==0);
    if (cr) {
        CHECK(cc[0]==17 && cc[1]==19);
        CHECK(ct[0]==23 && ct[1]==29 && ct[2]==31 && ct[3]==37);
    }
}
static void matrix(ColiSplitPlanner mojo) {
    for (unsigned i=0;i<10000;++i) {
        uint32_t c[4]={1,(uint32_t)(random_bits()%65),0,(uint32_t)(i&1)};
        c[2]=(uint32_t)(random_bits()%(c[1]+1));
        double cost[9];
        for (unsigned j=0;j<9;++j) {
            uint64_t bits=random_bits() & UINT64_C(0x7fefffffffffffff);
            if ((i+j)%13==0) bits=0;
            if ((i+j)%17==0) bits=random_bits() & UINT64_C(0xfffff);
            cost[j]=from_bits(bits);
        }
        compare(mojo,c,cost);
    }
}
#endif
int main(int argc,char **argv) {
    char error[512]; ColiMojoSplit *loaded=NULL;
    CHECK(coli_mojo_split_planner(NULL)==NULL);
    coli_mojo_split_close(NULL);
#ifdef _WIN32
    (void)argc;(void)argv;
    CHECK(coli_mojo_split_open("wsl-plugin.so",&loaded,error,sizeof error)!=0);
    CHECK(loaded==NULL && strstr(error,"WSL"));
    printf("%u native Windows Mojo refusal checks passed\n",checks);
    return 0;
#else
    CHECK(argc==5); /* real Mojo, wrong ABI, missing symbols, failed init */
    unsetenv("COLI_MOJO_DISPATCH");
    CHECK(coli_mojo_split_open(NULL,&loaded,error,sizeof error)!=0 && loaded==NULL);
    CHECK(coli_mojo_split_open("",&loaded,error,sizeof error)!=0 && loaded==NULL);
    CHECK(coli_mojo_split_open("/colibri/absent-mojo-library.so",&loaded,error,sizeof error)!=0 && loaded==NULL);
    CHECK(coli_mojo_split_open(argv[1],NULL,error,sizeof error)!=0);
    ColiMojoSplit *occupied=(ColiMojoSplit *)(uintptr_t)1;
    CHECK(coli_mojo_split_open(argv[1],&occupied,error,sizeof error)!=0 && occupied==(ColiMojoSplit *)(uintptr_t)1);
    for (unsigned i=2;i<5;++i) {
        CHECK(coli_mojo_split_open(argv[i],&loaded,error,sizeof error)!=0 && loaded==NULL);
        CHECK(error[0]!=0);
    }
    CHECK(setenv("COLI_MOJO_DISPATCH",argv[1],1)==0);
    CHECK(coli_mojo_split_open(NULL,&loaded,error,sizeof error)==0 && loaded && error[0]==0);
    ColiSplitPlanner mojo=coli_mojo_split_planner(loaded);
    CHECK(mojo!=NULL);
    uint32_t c[4]={1,8,8,1},counts[2];
    double cost[9]={10,2,3,0,0,0,0,0,0},times[4];
    CHECK(mojo(c,cost,counts,times)==0 && counts[0]==5 && counts[1]==3 && times[3]==30);
    compare(mojo,c,cost);
    c[3]=0;compare(mojo,c,cost);c[3]=1;
    CHECK(mojo(NULL,cost,counts,times)!=0);
    CHECK(mojo(c,NULL,counts,times)!=0);
    CHECK(mojo(c,cost,NULL,times)!=0);
    CHECK(mojo(c,cost,counts,NULL)!=0);
    for (unsigned k=0;k<4;++k) {
        uint32_t bad[4];memcpy(bad,c,sizeof bad);
        bad[k]=k==0?2:k==1?1025:k==2?9:2;
        compare(mojo,bad,cost);
    }
    const uint64_t malformed[]={UINT64_C(0x7ff0000000000000),UINT64_C(0xfff0000000000000),UINT64_C(0x7ff8000000000001),UINT64_C(0x7ff0000000000001),UINT64_C(0xbff0000000000000),UINT64_C(0x8000000000000001)};
    for (unsigned i=0;i<sizeof malformed/sizeof malformed[0];++i)
        for (unsigned j=0;j<9;++j) {
            double bad[9];memcpy(bad,cost,sizeof bad);bad[j]=from_bits(malformed[i]);compare(mojo,c,bad);
        }
    for (unsigned j=0;j<9;++j) {
        double edge[9];memcpy(edge,cost,sizeof edge);
        edge[j]=from_bits(UINT64_C(0x8000000000000000));compare(mojo,c,edge);
        edge[j]=DBL_MAX;compare(mojo,c,edge);
        edge[j]=from_bits(1);compare(mojo,c,edge);
    }
    double subnormal[9];for(unsigned i=0;i<9;++i)subnormal[i]=from_bits(1);
    compare(mojo,c,subnormal);
    uint32_t high[4]={1,1024,1024,1};compare(mojo,high,subnormal);
    uint32_t empty[4]={1,0,0,0};double zero[9]={0};compare(mojo,empty,zero);
    zero[0]=1;zero[2]=1;c[1]=2;c[2]=2;
    CHECK(mojo(c,zero,counts,times)==0 && counts[0]==1 && times[3]==1);
    zero[0]=2;zero[1]=0;zero[2]=2;c[1]=1;c[2]=1;
    CHECK(mojo(c,zero,counts,times)==0 && counts[0]==0); /* tie: fewer transfers */
    double overlap[5]={47,53,59,61,67},saved[5];memcpy(saved,overlap,sizeof saved);
    CHECK(mojo(c,cost,(uint32_t *)overlap,overlap)!=0 && memcmp(overlap,saved,sizeof saved)==0);
    CHECK(mojo(c,cost,(uint32_t *)(void *)(overlap+1),overlap)!=0 && memcmp(overlap,saved,sizeof saved)==0);
    CHECK(mojo(c,cost,(uint32_t *)(UINTPTR_MAX-3),times)!=0);
    CHECK(mojo(c,cost,counts,(double *)(UINTPTR_MAX-7))!=0);
    uint32_t alias_control[4]={1,8,8,1};double alias_cost[9];memcpy(alias_cost,cost,sizeof cost);
    CHECK(mojo(alias_control,alias_cost,alias_control,alias_cost+2)==0);
    CHECK(alias_control[0]==5 && alias_control[1]==3 && alias_cost[5]==30);
    matrix(mojo);
#if defined(__SSE__)
    unsigned mxcsr=_mm_getcsr();_mm_setcsr(mxcsr|0x8040); /* FTZ + DAZ */
    matrix(mojo);compare(mojo,high,subnormal);
    _mm_setcsr(mxcsr);
#endif
    coli_mojo_split_close(loaded);loaded=NULL;mojo=NULL;
    /* Repeat genuine load/init/run/close cycles, not stale borrowed calls. */
    for(unsigned i=0;i<20;++i) {
        CHECK(coli_mojo_split_open(argv[1],&loaded,error,sizeof error)==0);
        mojo=coli_mojo_split_planner(loaded);compare(mojo,high,subnormal);
        coli_mojo_split_close(loaded);loaded=NULL;
    }
    unsetenv("COLI_MOJO_DISPATCH");
    printf("%u native Mojo dispatch checks passed; %u exact C/Mojo planner comparisons\n",checks,comparisons);
    return 0;
#endif
}
#endif
