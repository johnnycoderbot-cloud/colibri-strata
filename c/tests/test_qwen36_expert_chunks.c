/* Real file-backed indexed slots: distinct-ID admission, partial/tail chunks,
 * low-cap rank cuts, negative route holes and balanced slot ownership. */
static void observe_chunk(void *,int,int,int,void *,int);
#ifdef COLI_CUDA
#undef COLI_CUDA
#endif
#define COLI_QWEN_XF_TEST_HOOK(m,r,k,u,h,n) observe_chunk((m),(r),(k),(u),(h),(n))
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include <assert.h>

enum { D=128,F=64,E=8,K=3 };
static int observed_runs,max_rows,observed_pairs;

static void observe_chunk(void *model,int rows,int kper,int distinct,void *raw,int count){
    Model *m=model;Slot **held=raw;LCache *lc=&m->cache[0];
    int unique=0,ids[E]={0};
    assert(rows>=1&&rows<=128&&count==rows*kper);
    for(int i=0;i<count;i++)if(held[i]){
        assert(slot_busy(held[i]));assert(held[i]->eid>=0&&held[i]->eid<E);
        if(!ids[held[i]->eid]){ids[held[i]->eid]=1;unique++;}
        for(int j=0;j<i;j++)if(held[j]&&held[j]->eid==held[i]->eid)assert(held[j]==held[i]);
    }
    assert(unique<=lc->cap&&lc->n<=lc->cap);
    if(rows>1)assert(distinct==unique);
    pthread_mutex_lock(&g_pilot_mx);
    int victim=slot_victim(lc,0,1);
    if(victim>=0)assert(!slot_busy(&lc->slots[victim]));
    pthread_mutex_unlock(&g_pilot_mx);
    observed_runs++;observed_pairs+=count;if(rows>max_rows)max_rows=rows;
}

static FILE *fixture(Model *m,int cap){
    memset(m,0,sizeof(*m));m->c.hidden=D;m->c.inter=F;
    m->c.n_experts=E;m->c.topk=K;m->c.n_layers=1;m->c.expert_gs=64;
    m->active_of=calloc(1,sizeof(int));m->is_pinned=calloc(E,1);
    m->cache=calloc(1,sizeof(LCache));LCache *lc=&m->cache[0];
    lc->cap=cap;lc->slots=calloc((size_t)cap,sizeof(Slot));
    lc->slot_by_expert=malloc(E*sizeof(int));assert(lc->slots&&lc->slot_by_expert);
    for(int e=0;e<E;e++)lc->slot_by_expert[e]=-1;
    FILE *file=tmpfile();assert(file);int fd=fileno(file);
    m->S.n=E*2;m->S.t=calloc(E*2,sizeof(st_tensor));m->S.nfd=1;m->S.fds[0]=fd;
    int bytes=3*D*F/2,nscale=2*F*(D/64)+D*(F/64);
    uint8_t *weights=malloc((size_t)bytes);float *sc=falloc(nscale);assert(weights);
    for(int e=0;e<E;e++){
        st_tensor *tw=&m->S.t[e*2],*ts=&m->S.t[e*2+1];char name[160];
        snprintf(name,sizeof(name),"model.layers.0.mlp.experts.%d.merged_weight",e);tw->name=strdup(name);
        tw->fd=fd;tw->off=ftell(file);tw->nbytes=tw->numel=bytes;tw->dtype=3;
        for(int i=0;i<bytes;i++)weights[i]=(uint8_t)(i*37+e*13);
        assert(fwrite(weights,1,(size_t)bytes,file)==(size_t)bytes);
        snprintf(name,sizeof(name),"model.layers.0.mlp.experts.%d.qs",e);ts->name=strdup(name);
        ts->fd=fd;ts->off=ftell(file);ts->numel=nscale;ts->nbytes=nscale*sizeof(float);ts->dtype=2;
        for(int i=0;i<nscale;i++)sc[i]=0.003f+(float)((i*19+e)%17)*0.001f;
        assert(fwrite(sc,sizeof(float),(size_t)nscale,file)==(size_t)nscale);
    }
    assert(!fflush(file));m->S.sizes[0]=ftell(file);free(weights);free(sc);return file;
}

static void destroy(Model *m,FILE *file){
    LCache *lc=&m->cache[0];
    for(int i=0;i<lc->n;i++){assert(!slot_busy(&lc->slots[i]));free(lc->slots[i].pw);free(lc->slots[i].gs);}
    for(int i=0;i<m->S.n;i++)free(m->S.t[i].name);
    free(m->S.t);free(lc->slots);free(lc->slot_by_expert);free(m->cache);
    free(m->active_of);free(m->is_pinned);
    if(m->ehit){free(m->ehit[0]);free(m->ehit);}fclose(file);
}

/* Independent per-(row,rank) reference: each dot is the shared scalar/SIMD
 * kernel, but scheduling and cache ownership never use moe_xf_run. */
static void reference(Model *m,const float *x,int S,float *out,const int *idx,const float *val){
    void *scratch=malloc(xf_moe_scratch_bytes(1,1,D,F));assert(scratch);
    const int64_t gp=(int64_t)D*F/2;
    for(int s=0;s<S;s++)for(int k=0;k<K;k++){
        int i=s*K+k;if(idx[i]<0)continue;
        Slot *e=expert_hold(m,0,idx[i]);
        XfExpert expert={e->pw,e->pw+gp,e->pw+2*gp,e->gs,e->us,e->ds};
        const XfExpert *p=&expert;
        xf_moe_add(out+(int64_t)s*D,x+(int64_t)s*D,1,1,D,F,idx+i,val+i,&p,xf_act_mode(),scratch);
        slot_release(e);
    }
    free(scratch);
}

static void one_case(int S,int cap,int repeated,int holes){
    Model a,b;FILE *fa=fixture(&a,cap),*fb=fixture(&b,cap);
    float *x=falloc((int64_t)S*D),*seed=falloc((int64_t)S*D);
    float *ref=falloc((int64_t)S*D),*out=falloc((int64_t)S*D),*val=falloc((int64_t)S*K);
    int *idx=malloc((size_t)S*K*sizeof(int));assert(idx);int valid=0;
    for(int64_t i=0;i<(int64_t)S*D;i++){x[i]=(float)((int)((i*31)%251)-125)/101.f;seed[i]=(float)(i%13)/97.f;}
    for(int s=0;s<S;s++)for(int k=0;k<K;k++){
        int i=s*K+k;idx[i]=repeated?k:(s+k)%E;if(holes&&i%7==0)idx[i]=-1;
        val[i]=(float)(k+1)/7.f;if(idx[i]>=0)valid++;
    }
    memcpy(ref,seed,(size_t)S*D*sizeof(float));memcpy(out,seed,(size_t)S*D*sizeof(float));
    reference(&a,x,S,ref,idx,val);
    observed_runs=max_rows=observed_pairs=0;
    moe_xf_run(&b,0,x,S,out,idx,val);
    assert(!memcmp(ref,out,(size_t)S*D*sizeof(float)));
    assert(a.hits+a.miss==(uint64_t)valid&&b.hits+b.miss==(uint64_t)valid);
    assert(observed_pairs==S*K);
    if(cap<K)assert(max_rows==1&&observed_runs==S*K);
    if(repeated&&cap>=K&&S>1){assert(max_rows==(S<128?S:128));assert(observed_runs==(S+127)/128);}
    if(!repeated&&cap==K&&S>1)assert(max_rows==1);
    printf("expert chunks exact: S=%d cap=%d repeat=%d holes=%d runs=%d max_rows=%d misses=%llu/%llu\n",
           S,cap,repeated,holes,observed_runs,max_rows,(unsigned long long)a.miss,(unsigned long long)b.miss);
    destroy(&a,fa);destroy(&b,fb);free(x);free(seed);free(ref);free(out);free(idx);free(val);
}

int main(int argc,char **argv){
    setenv("QWEN_EXPERT_ACT",argc>1&&argv[1][0]=='1'?"i8":"f32",1);
    unsetenv("QWEN_EXPERT_BATCH");setenv("QWEN_EXPERT_KERNEL","1",1);g_expert_gs=64;
    one_case(1,1,0,0);one_case(7,1,0,1);one_case(17,K,0,0);
    one_case(17,4,0,1);one_case(19,E,0,0);one_case(129,E,1,0);
    one_case(133,K,1,1);
    puts("qwen36 expert chunks: ok");return 0;
}
