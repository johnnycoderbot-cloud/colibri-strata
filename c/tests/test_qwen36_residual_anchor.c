/* Raw residual anchor ownership/position regression. The zero-layer fixture
 * executes real step_ex and speculative acceptance/rollback bookkeeping.
 * It measures numerical correctness only, never model throughput. */
#ifdef COLI_CUDA
#undef COLI_CUDA
#endif
#ifdef COLI_VULKAN
#undef COLI_VULKAN
#endif
#define COLI_STRATA_NATIVE 1
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main
#include <assert.h>

enum { D=64, V=259 };
static void fixture_init(Model *m) {
    memset(m,0,sizeof(*m));
    m->c.hidden=D; m->c.vocab=V; m->c.eps=1e-6f; m->kv_cap=64;
    m->embed=falloc((int64_t)V*D); m->final_norm=falloc(D);
    m->lm_head.I=D; m->lm_head.O=V;
    float *weight=falloc((int64_t)V*D); m->lm_head.w=weight;
    for(int i=0;i<D;i++)m->final_norm[i]=1.f+(float)(i%7)*.01f;
    for(int64_t i=0;i<(int64_t)V*D;i++) {
        m->embed[i]=(float)((int)((i*37+13)%251)-125)/99.f;
        weight[i]=(float)((int)((i*19+71)%239)-119)/103.f;
    }
    qw_quantize(m->lm_head.w,D,V,NULL,&m->lm_head);
}
static void model_free(Model *m) {
    free(m->embed); free(m->final_norm); qw_free(&m->lm_head);
    for(int sl=0;sl<m->snap_slots;sl++) { free(m->snap_rec[sl]); free(m->snap_conv[sl]); }
    kv_prefix_free(&m->kvp);
}
static int top1(const float *logit) {
    int best=0; for(int v=1;v<V;v++)if(logit[v]>logit[best])best=v; return best;
}
static void check_raw(const float *logit,int pos) {
    assert(g_sn_valid && g_sn_pos==pos);
    assert(g_sn_base_valid && g_sn_base_pos==pos);
    assert(!memcmp(g_sn_base,logit,V*sizeof(float)));
}
static void publication(void) {
    Model m; fixture_init(&m); int ids[6]={3,17,29,43,61,79};
    for(int S=1;S<=6;S++) {
        g_sn_capture_rows=0;
        float *logit=step_ex(&m,ids,S,7,1);
        check_raw(logit,7+S-1);
        logit[0]+=12345.f; assert(logit[0]!=g_sn_base[0]); free(logit);
        g_sn_capture_rows=S;
        logit=step_ex(&m,ids,S,7,S);
        /* The verifier commits row zero after returning from step_ex. */
        memcpy(g_sn_frontier,g_sn_rows,D*sizeof(float));
        g_sn_valid=1; g_sn_pos=7;
        check_raw(logit,7);
        logit[0]+=12345.f; assert(logit[0]!=g_sn_base[0]); free(logit);
    }
    g_sn_capture_rows=3; g_sn_base_valid=1;
    float dummy[V]={0}; q36_sn_publish_logits(&m,dummy,3,1,7);
    assert(!g_sn_base_valid); g_sn_capture_rows=0;
    model_free(&m);
}
static void speculative(void) {
    Model m,oracle; fixture_init(&m); fixture_init(&oracle);
    int ids[24];ids[0]=3;
    for(int p=0;p<23;p++) {
        float *logit=step(&oracle,ids+p,1,p); ids[p+1]=top1(logit); free(logit);
    }
    g_q36_oracle=ids;g_q36_oracle_n=24;
    const char *modes[]={"accept","row1","row2"};
    for(int mode=0;mode<3;mode++) {
        setenv("COLI_LOOKUP_FORCE",modes[mode],1);
        Q36Spec sp;q36_spec_begin(&m,&sp,ids,1);
        assert(!g_sn_valid && !g_sn_base_valid);
        float *logit=step(&m,ids,1,0);check_raw(logit,0); free(logit);
        for(int p=1;p<12;p++) {
            logit=q36_spec_step(&m,&sp,ids[p],p,24-p);check_raw(logit,p);
            logit[0]+=12345.f;assert(logit[0]!=g_sn_base[0]);free(logit);
        }
        assert(sp.verifies>0);
        if(mode==0)assert(sp.accepted>0);
        q36_spec_end(&m,&sp);assert(!g_sn_valid && !g_sn_base_valid);
        q36_spec_free(&sp);
    }
    unsetenv("COLI_LOOKUP_FORCE");model_free(&oracle);model_free(&m);
}
int main(void) {
    setenv("COLI_LOOKUP","1",1);setenv("COLI_SPEC_GATE","0",1);
    setenv("COLI_DENSE_IDOT","1",1);setenv("COLI_DENSE_BITS","8",1);
    g_sn_h=D;g_sn_residual=1;
    g_sn_frontier=falloc(D);g_sn_rows=falloc(6*D);g_sn_base=falloc(V);
    publication();speculative();q36_sn_shutdown();
    puts("residual anchor: raw ownership, prefill/verify rows, acceptance/rejection and reset exact");
    return 0;
}
