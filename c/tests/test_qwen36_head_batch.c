/* Isolate step_ex's real output projection with a zero-layer in-memory model.
 * Compare every returned row to independently normalized one-row projections,
 * and count the matmul_d calls. This is a numerical/profile gate, not a model
 * throughput measurement. Run once with argument 0 (F32 activations), once 1. */
#define COLI_QWEN_BATCH_TEST 1
#ifdef COLI_CUDA
#undef COLI_CUDA
#endif
#ifdef COLI_VULKAN
#undef COLI_VULKAN
#endif
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include <assert.h>

static float value(int64_t i, int salt) {
    return (float)((int)((i*37+salt*13)%251)-125)/(float)(97+salt);
}

static void one_format(int format, int S, int nlogits) {
    enum { D=64,V=259 };
    Model m; memset(&m,0,sizeof(m));
    m.c.hidden=D; m.c.vocab=V; m.c.eps=1e-6f;
    m.embed=falloc((int64_t)V*D); m.final_norm=falloc(D);
    for(int64_t i=0;i<(int64_t)V*D;i++) m.embed[i]=value(i,1);
    for(int i=0;i<D;i++)m.final_norm[i]=0.8f+(float)(i%13)*0.01f;
    m.lm_head.I=D;m.lm_head.O=V;
    if(format==1){
        m.lm_head.h=malloc((size_t)V*D*sizeof(uint16_t));assert(m.lm_head.h);
        for(int64_t i=0;i<(int64_t)V*D;i++)m.lm_head.h[i]=(uint16_t)(0x3800+(i*17)%0x800);
    }else{
        float *w=falloc((int64_t)V*D);
        for(int64_t i=0;i<(int64_t)V*D;i++)w[i]=value(i,2);
        m.lm_head.w=w;
        if(format==2)qw_quantize(w,D,V,NULL,&m.lm_head);
        if(format==3){
            m.lm_head.q4=malloc((size_t)V*D/2);m.lm_head.sg=falloc(V);
            assert(m.lm_head.q4);pack_int4_g64_planar(w,m.lm_head.q4,m.lm_head.sg,V,D);
            m.lm_head.ng=1;
        }
    }
    int ids[17];for(int i=0;i<S;i++)ids[i]=(i*11+3)%V;
    float *want=falloc((int64_t)nlogits*V),row[D];
    for(int r=0;r<nlogits;r++){
        rmsnorm_row(row,m.embed+(int64_t)ids[S-nlogits+r]*D,m.final_norm,D,m.c.eps);
        matmul_d(want+(int64_t)r*V,row,&m.lm_head,1,D,V);
    }
    setenv("QWEN_HEAD_BATCH","0",1);g_qwen_matmul_d_calls=0;
    float *legacy=step_ex(&m,ids,S,0,nlogits);
    assert(g_qwen_matmul_d_calls==(uint64_t)nlogits);
    assert(!memcmp(want,legacy,(size_t)nlogits*V*sizeof(float)));
    unsetenv("QWEN_HEAD_BATCH");g_qwen_matmul_d_calls=0;
    float *got=step_ex(&m,ids,S,0,nlogits);
    uint64_t calls=format==3?(uint64_t)nlogits:1;
    assert(g_qwen_matmul_d_calls==calls);
    assert(!memcmp(want,got,(size_t)nlogits*V*sizeof(float)));
    printf("head rows exact: format=%d S=%d logits=%d projection_calls=%d -> %llu\n",
           format,S,nlogits,nlogits,(unsigned long long)calls);
    free(want);free(legacy);free(got);free(m.embed);free(m.final_norm);qw_free(&m.lm_head);
}

int main(int argc,char **argv){
    setenv("COLI_DENSE_IDOT",argc>1&&argv[1][0]=='1'?"1":"0",1);
    setenv("COLI_DENSE_BITS","8",1);setenv("COLI_DENSE_I8","1",1);
    for(int format=0;format<4;format++){
        one_format(format,1,1);one_format(format,5,2);
        one_format(format,9,9);one_format(format,17,7);
    }
    puts("qwen36 head batch: ok");return 0;
}
