/* Actual MAX C SDK replay; fixture is a synthetic projection/relu graph. */
#define _POSIX_C_SOURCE 200809L
#include "../native_mojo_backend.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks;
static void check(int yes){checks++;assert(yes);}
int main(int argc,char **argv) {
    if(argc!=2){fputs("test_native_mojo_backend model.mef\n",stderr);return 2;}
#ifdef COLI_MOJO_CONFIG_DIR
    setenv("MODULAR_HOME",COLI_MOJO_CONFIG_DIR,1);
#endif
    CMJEngine *engine=NULL;char error[4096];
    for(unsigned d=CMJ_F32;d<=CMJ_BF16;d++) {
        unsigned bits=d==CMJ_I4?4:(d==CMJ_I8||d==CMJ_U8)?8:(d==CMJ_I16||d==CMJ_F16||d==CMJ_BF16)?16:d==CMJ_I64?64:32;
        int64_t shape=(int64_t)(CMJ_MAX_BYTES*8/bits);size_t bytes=0;
        check(cmj_tensor_bytes((CMJDType)d,1,&shape,&bytes)==CMJ_OK&&bytes==CMJ_MAX_BYTES);
        shape++;bytes=123;
        check(cmj_tensor_bytes((CMJDType)d,1,&shape,&bytes)==CMJ_INVALID&&bytes==123);
    }
    CMJOptions missing={cmj_accelerator_count(),argv[1],128,NULL,0};
    check(cmj_create(&missing,&engine,error,sizeof error)==CMJ_DEVICE&&!engine);
    CMJOptions opts={-1,argv[1],128,NULL,0};
    check(cmj_create(&opts,&engine,error,sizeof error)==CMJ_OK&&engine);
    float x[]={1,-2,3,4,5,-6},w[]={2,1,-1,3,4,-2,2,1,-1,3,5,-2};
    const float expected[]={0,14,10,0,34,0,0,29};float saved[6];memcpy(saved,x,sizeof x);
    CMJInput input[2]={{"input0",CMJ_F32,2,{2,3},x,sizeof x},{"input1",CMJ_F32,2,{3,4},w,sizeof w}};
    const char *names[]={"output0"};CMJOutput out[2]={0};
    for(unsigned k=0;k<10;k++) {
        check(cmj_run(engine,input,2,names,1,out,error,sizeof error)==CMJ_OK);
        check(out[0].bytes==sizeof expected&&out[0].dtype==CMJ_F32&&!out[0].source_accelerator&&out[0].source_device_id==0);
        check(!memcmp(out[0].data,expected,sizeof expected)&&!memcmp(x,saved,sizeof x));
        cmj_output_free(out,1);
    }
    out[0].data=x;
    check(cmj_run(engine,input,2,names,1,out,error,sizeof error)==CMJ_INVALID&&out[0].data==x);
    out[0].data=NULL;
    unsigned char canary[sizeof out];memset(out,0x5a,sizeof out);out[0].data=NULL;
    memcpy(canary,out,sizeof out);
    input[0].bytes--;check(cmj_run(engine,input,2,names,1,out,error,sizeof error)==CMJ_INVALID);
    check(!memcmp(out,canary,sizeof out));input[0].bytes++;
    x[0]=NAN;check(cmj_run(engine,input,2,names,1,out,error,sizeof error)==CMJ_INVALID);x[0]=saved[0];
    input[0].shape[0]=0;check(cmj_run(engine,input,2,names,1,out,error,sizeof error)==CMJ_INVALID);input[0].shape[0]=2;
    input[0].shape[0]=1;input[0].bytes=3*sizeof(float);
    check(cmj_run(engine,input,2,names,1,out,error,sizeof error)!=CMJ_OK);
    input[0].shape[0]=2;input[0].bytes=sizeof x;
    input[1].name="input0";check(cmj_run(engine,input,2,names,1,out,error,sizeof error)==CMJ_INVALID);input[1].name="input1";
    const char *bad_names[]={"missing-output"};check(cmj_run(engine,input,2,bad_names,1,out,error,sizeof error)!=CMJ_OK);
    check(!memcmp(out,canary,sizeof out));
    const char *duplicate[]={"output0","output0"};memset(out,0,sizeof out);
    check(cmj_run(engine,input,2,duplicate,2,out,error,sizeof error)==CMJ_INVALID);
    input[0].dtype=(CMJDType)100;check(cmj_run(engine,input,2,names,1,out,error,sizeof error)==CMJ_INVALID);input[0].dtype=CMJ_F32;
    input[0].dtype=CMJ_I32;
    check(cmj_run(engine,input,2,names,1,out,error,sizeof error)!=CMJ_OK&&!out[0].data);
    input[0].dtype=CMJ_F32;
    input[0].rank=CMJ_MAX_RANK+1;check(cmj_run(engine,input,2,names,1,out,error,sizeof error)==CMJ_INVALID);input[0].rank=2;
    opts.result_limit_bytes=1;CMJEngine *small=NULL;
    check(cmj_create(&opts,&small,error,sizeof error)==CMJ_OK&&small);
    check(cmj_run(small,input,2,names,1,out,error,sizeof error)==CMJ_LIMIT&&!out[0].data);
    cmj_destroy(small);cmj_destroy(engine);
    printf("native Mojo/MAX C: %u actual SDK checks passed; exact CPU fixture; no physical GPU/NPU/TPU/AIPU speed claim\n",checks);
    return 0;
}
