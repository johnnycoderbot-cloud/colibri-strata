/* Native MTP frontier ownership tests. A real nontrivial cell drafts against a
 * zero-layer target; forced oracle IDs exercise bookkeeping only, never speed.
 * Reference target logits are computed before installing the global drafter. */
#ifdef COLI_CUDA
#undef COLI_CUDA
#endif
#ifdef COLI_VULKAN
#undef COLI_VULKAN
#endif
#define COLI_STRATA_NATIVE 1
#define COLI_QWEN_MTP_TEST 1
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main
#include <assert.h>

enum { H=16, V=37, QH=4, KH=2, HD=8, F=24, NT=96, PROMPT=5, KVROW=KH*HD };
static int oracle_ids[NT];
static float oracle_logits[NT][V], oracle_hidden[NT][H];

static float fixture_value(size_t i, int salt) {
    return (float)((int)((i*37+(size_t)salt*19)%127)-63)/(float)(79+salt*3);
}
static void target_init(Model *m,int capacity) {
    assert(!g_q36_mtp);
    memset(m,0,sizeof(*m));
    m->c.hidden=H;m->c.vocab=V;m->c.eps=1e-6f;
    m->c.q_heads=QH;m->c.kv_heads=KH;m->c.head_dim=HD;
    m->c.q_head_dim=2*HD;m->c.k_head_dim=m->c.v_head_dim=HD;
    m->c.o_in=QH*HD;m->c.rope_dim=m->c.rotary_dim=HD;
    m->c.theta=10000.f;m->c.partial_rotary_factor=1.f;
    m->c.shared_inter=F;m->c.zero_centered_norms=1;
    m->c.attn_output_gate=m->c.has_qk_norm=1;m->c.dn_convk=1;
    m->c.mrope_section[0]=2;m->c.mrope_section[1]=m->c.mrope_section[2]=1;
    m->embed=falloc((int64_t)V*H);m->final_norm=falloc(H);
    m->lm_head.I=H;m->lm_head.O=V;
    float *weights=falloc((int64_t)V*H);m->lm_head.w=weights;
    for(int i=0;i<H;i++)m->final_norm[i]=.13f+(float)(i%7)*.017f;
    for(int i=0;i<V*H;i++){m->embed[i]=fixture_value((size_t)i,1);weights[i]=fixture_value((size_t)i,3);}
    qw_quantize(weights,H,V,NULL,&m->lm_head);
    m->max_t=capacity;ensure_kv(m);
}
static void target_free(Model *m) {
    free(m->embed);free(m->final_norm);qw_free(&m->lm_head);
    free(m->K);free(m->V);free(m->attn_sc);
    for(int slot=0;slot<m->snap_slots;slot++){free(m->snap_rec[slot]);free(m->snap_conv[slot]);}
    kv_prefix_free(&m->kvp);
}
static int best_token(const float *logit) {
    int best=0;for(int v=1;v<V;v++)if(logit[v]>logit[best])best=v;return best;
}
static void make_oracle(void) {
    Model m;target_init(&m,NT);
    for(int p=0;p<PROMPT;p++)oracle_ids[p]=(p*7+3)%V;
    for(int p=0;p<NT;p++){
        float *logit=step(&m,oracle_ids+p,1,p);
        memcpy(oracle_logits[p],logit,V*sizeof(float));free(logit);
        rmsnorm_row(oracle_hidden[p],m.embed+(size_t)oracle_ids[p]*H,m.final_norm,H,m.c.eps);
        if(p>=PROMPT-1&&p+1<NT)oracle_ids[p+1]=best_token(oracle_logits[p]);
    }
    g_q36_oracle=oracle_ids;g_q36_oracle_n=NT;target_free(&m);
}

static Q36Mtp *cell_new(Model *m,int bits,int capacity) {
    const size_t counts[Q36_MTP_TENSOR_COUNT]={
        H*2*H,H,H,H,2*QH*HD*H,KH*HD*H,KH*HD*H,H*QH*HD,HD,HD,H,F*H,F*H,H*F,H
    };
    Q36MtpWeights weights;memset(&weights,0,sizeof weights);
    float *owned[Q36_MTP_TENSOR_COUNT];
    for(int t=0;t<Q36_MTP_TENSOR_COUNT;t++){
        owned[t]=falloc((int64_t)counts[t]);weights.values[t]=owned[t];weights.counts[t]=counts[t];
        for(size_t i=0;i<counts[t];i++)owned[t][i]=fixture_value(i,t+5)*.17f;
    }
    Q36Mtp *cell=q36_mtp_from_weights(m,&weights,capacity,bits);assert(cell);
    for(int t=0;t<Q36_MTP_TENSOR_COUNT;t++)free(owned[t]);
    return cell;
}
static void install(Model *m,int bits,int capacity) {
    assert(!g_q36_mtp);g_q36_mtp=cell_new(m,bits,capacity);g_mp_h=H;g_mp_depth=3;
    g_mp_pending=falloc(Q36_SPEC_ROWS*H);g_mp_rows=falloc(Q36_SPEC_ROWS*H);
    g_mp_chunk=falloc(32*H);g_mp_chain=falloc(2*H);g_mp_logit=falloc(V);
    g_mp_oracle=NULL;g_mp_oracle_n=g_mp_oracle_prompt=0;
    g_mp_force=g_mp_force_row=0;memset(g_mp_next,0,sizeof g_mp_next);q36_mp_reset();
}
static void close_active(void) {
    pin_drop();
    for(int i=0;i<g_pins.n;i++){free(g_pins.slot[i].ids);free(g_pins.slot[i].logit);}
    memset(&g_pins,0,sizeof g_pins);
    q36_mp_shutdown();g_mp_force=g_mp_force_row=0;
}
static void peek(Q36Mtp *cell,int position,float key[KVROW],float value[KVROW]) {
    assert(q36_mtp_test_peek_allocated(cell,position,key,value));
}
static void same_prefix(Q36Mtp *left,Q36Mtp *right,int length) {
    assert(q36_mtp_length(left)>=length&&q36_mtp_length(right)>=length);
    for(int p=0;p<length;p++){
        float ak[KVROW],av[KVROW],bk[KVROW],bv[KVROW];peek(left,p,ak,av);peek(right,p,bk,bv);
        assert(!memcmp(ak,bk,sizeof ak));assert(!memcmp(av,bv,sizeof av));
    }
}
static Q36Mtp *teacher(Model *m,int bits,int rows,int capacity) {
    Q36Mtp *cell=cell_new(m,bits,capacity);
    for(int pos=0;pos<rows;){
        int batch=rows-pos<Q36_MTP_BATCH?rows-pos:Q36_MTP_BATCH;
        assert(q36_mtp_kv_rows(cell,&oracle_hidden[pos][0],oracle_ids+pos+1,batch,pos));pos+=batch;
    }
    assert(q36_mtp_length(cell)==rows);return cell;
}
static void pending_matches(int target_rows) {
    assert(g_mp_len>=0&&g_mp_pending_n>=1&&g_mp_pending_n<=Q36_SPEC_ROWS);
    assert(g_mp_len+g_mp_pending_n==target_rows);
    assert(q36_mtp_length(g_q36_mtp)==g_mp_len);
    assert(!memcmp(g_mp_pending,&oracle_hidden[g_mp_len][0],(size_t)g_mp_pending_n*H*sizeof(float)));
    for(int j=0;j+1<g_mp_pending_n;j++)assert(g_mp_next[j]==oracle_ids[g_mp_len+j+1]);
}
static void final_teacher_equivalence(Model *m,int bits,int target_rows) {
    pending_matches(target_rows);
    Q36Mtp *reference=teacher(m,bits,target_rows,q36_mtp_capacity(g_q36_mtp));
    same_prefix(g_q36_mtp,reference,g_mp_len);
    /* Finish only known actual target pairs, including the next token selected
     * by the last actual target distribution. Predicted hidden rows are unused. */
    assert(q36_mp_take(oracle_ids[target_rows],NULL,NULL));
    assert(g_mp_pending_n==0&&g_mp_len==target_rows);
    same_prefix(g_q36_mtp,reference,target_rows);q36_mtp_free(reference);
}

static void prefill_equivalence(int bits) {
    enum {N=35,CAP=64};Model a,b;target_init(&a,CAP);target_init(&b,CAP);
    install(&a,bits,CAP);float *logit=step(&a,oracle_ids,N,0);
    assert(!memcmp(logit,oracle_logits[N-1],V*sizeof(float)));free(logit);pending_matches(N);
    Q36Mtp *full=g_q36_mtp;g_q36_mtp=NULL;
    float pending[H];memcpy(pending,g_mp_pending,sizeof pending);
    free(g_mp_pending);free(g_mp_rows);free(g_mp_chunk);free(g_mp_chain);free(g_mp_logit);
    g_mp_pending=g_mp_rows=g_mp_chunk=g_mp_chain=g_mp_logit=NULL;
    install(&b,bits,CAP);
    const int chunks[]={1,2,5,11,16};int pos=0;
    for(size_t i=0;i<sizeof chunks/sizeof chunks[0];i++){
        int rows=chunks[i];logit=step(&b,oracle_ids+pos,rows,pos);
        assert(!memcmp(logit,oracle_logits[pos+rows-1],V*sizeof(float)));free(logit);pos+=rows;
    }
    assert(pos==N);pending_matches(N);assert(!memcmp(pending,g_mp_pending,sizeof pending));
    assert(q36_mtp_length(full)==N-1);same_prefix(full,g_q36_mtp,N-1);
    Q36Mtp *reference=teacher(&b,bits,N-1,CAP);same_prefix(g_q36_mtp,reference,N-1);
    q36_mtp_free(reference);q36_mtp_free(full);close_active();target_free(&a);target_free(&b);
}

static void speculative_mode(int bits,int force,int wrong_row,int end_at) {
    Model m;target_init(&m,64);install(&m,bits,64);g_mp_force=force;g_mp_force_row=wrong_row;
    Q36Spec sp;q36_spec_begin(&m,&sp,oracle_ids,PROMPT);
    float *logit=step(&m,oracle_ids,PROMPT,0);free(logit);
    for(int pos=PROMPT;pos<end_at;pos++){
        logit=q36_spec_step(&m,&sp,oracle_ids[pos],pos,NT-pos);
        assert(!memcmp(logit,oracle_logits[pos],V*sizeof(float)));free(logit);
    }
    assert(sp.verifies>0&&sp.mtp_calls>0&&sp.mtp_failures==0);
    if(force=='a')assert(sp.accepted>0);
    if(force=='w'&&wrong_row==1)assert(sp.accepted==0);
    if(force=='w'&&wrong_row==2)assert(sp.accepted>0&&sp.accepted<sp.drafts);
    if(force=='c')assert(sp.accepted>0&&sp.accepted<sp.drafts);
    q36_spec_end(&m,&sp);assert(!sp.ahead_n&&!sp.ahead_logit&&!sp.ahead_mtp);
    assert(m.kv_len==end_at&&m.kvp.len==end_at);
    assert(kv_prefix_holds(&m.kvp,oracle_ids,end_at));
    final_teacher_equivalence(&m,bits,end_at);q36_spec_free(&sp);
    close_active();target_free(&m);
}

static void predicted_tail_is_replayed(int bits) {
    Model m;target_init(&m,32);install(&m,bits,32);g_mp_force='a';
    Q36Spec sp;q36_spec_begin(&m,&sp,oracle_ids,PROMPT);
    float *logit=step(&m,oracle_ids,PROMPT,0);free(logit);
    logit=q36_spec_step(&m,&sp,oracle_ids[PROMPT],PROMPT,20);free(logit);
    assert(sp.ahead_n==4&&g_mp_len==PROMPT&&g_mp_pending_n==0);
    float predicted_key[KVROW],predicted_value[KVROW];
    peek(g_q36_mtp,PROMPT,predicted_key,predicted_value);
    Q36Mtp *reference=teacher(&m,bits,PROMPT+4,32);
    float actual_key[KVROW],actual_value[KVROW];peek(reference,PROMPT,actual_key,actual_value);
    /* Nontrivial cell ensures this is a test of hidden/KV ownership rather
     * than a degenerate equality of token IDs or all-zero key/value tensors. */
    assert(memcmp(predicted_key,actual_key,sizeof actual_key)||memcmp(predicted_value,actual_value,sizeof actual_value));
    for(int pos=PROMPT+1;pos<=PROMPT+3;pos++){
        logit=q36_spec_step(&m,&sp,oracle_ids[pos],pos,20);
        assert(!memcmp(logit,oracle_logits[pos],V*sizeof(float)));free(logit);
    }
    assert(sp.accepted==3&&!sp.ahead_n);pending_matches(PROMPT+4);
    assert(q36_mp_take(oracle_ids[PROMPT+4],NULL,NULL));
    same_prefix(g_q36_mtp,reference,PROMPT+4);
    float replay_key[KVROW],replay_value[KVROW];peek(g_q36_mtp,PROMPT,replay_key,replay_value);
    assert(!memcmp(actual_key,replay_key,sizeof actual_key)&&!memcmp(actual_value,replay_value,sizeof actual_value));
    q36_spec_end(&m,&sp);q36_spec_free(&sp);q36_mtp_free(reference);close_active();target_free(&m);
}

static void cancellation_and_budget(int bits) {
    Model m;target_init(&m,32);install(&m,bits,32);g_mp_force='a';
    Q36Spec sp;q36_spec_begin(&m,&sp,oracle_ids,PROMPT);
    float *logit=step(&m,oracle_ids,PROMPT,0);free(logit);
    logit=q36_spec_step(&m,&sp,oracle_ids[PROMPT],PROMPT,20);free(logit);
    assert(sp.ahead_n==4);q36_spec_end(&m,&sp);assert(sp.ended_ahead);
    assert(m.kv_len==PROMPT+1&&m.kvp.len==PROMPT+1);
    final_teacher_equivalence(&m,bits,PROMPT+1);q36_spec_free(&sp);
    reset_recurrent(&m);assert(g_mp_len==0&&g_mp_pending_n==0&&q36_mtp_length(g_q36_mtp)==0);
    assert(!q36_mp_ready(PROMPT));close_active();target_free(&m);

    const int capacities[]={32,PROMPT+1};
    for(int c=0;c<2;c++){
        target_init(&m,capacities[c]);install(&m,bits,capacities[c]);g_mp_force='a';
        q36_spec_begin(&m,&sp,oracle_ids,PROMPT);logit=step(&m,oracle_ids,PROMPT,0);free(logit);
        logit=q36_spec_step(&m,&sp,oracle_ids[PROMPT],PROMPT,c==0?1:20);
        assert(!memcmp(logit,oracle_logits[PROMPT],V*sizeof(float)));free(logit);
        assert(sp.verifies==0&&sp.mtp_calls==0&&!sp.ahead_n);
        q36_spec_end(&m,&sp);final_teacher_equivalence(&m,bits,PROMPT+1);
        q36_spec_free(&sp);close_active();target_free(&m);
    }
}

static void pin_and_growth(int bits) {
    Model m;target_init(&m,8);install(&m,bits,8);
    float *logit=step(&m,oracle_ids,PROMPT,0);pin_save(&m,oracle_ids,PROMPT,logit);free(logit);
    pending_matches(PROMPT);Q36Mtp *reference=teacher(&m,bits,PROMPT-1,48);
    same_prefix(g_q36_mtp,reference,PROMPT-1);
    int old_capacity=q36_mtp_capacity(g_q36_mtp);m.max_t=48;ensure_kv(&m);
    assert(q36_mtp_capacity(g_q36_mtp)>=48&&old_capacity<48);
    same_prefix(g_q36_mtp,reference,PROMPT-1);pending_matches(PROMPT);
    for(int pos=PROMPT;pos<12;pos++){logit=step(&m,oracle_ids+pos,1,pos);free(logit);}
    assert(pin_restore(&m,oracle_ids,12)==PROMPT);pending_matches(PROMPT);
    assert(g_pin_logit&&!memcmp(g_pin_logit,oracle_logits[PROMPT-1],V*sizeof(float)));
    same_prefix(g_q36_mtp,reference,PROMPT-1);
    logit=step(&m,oracle_ids+PROMPT,1,PROMPT);
    assert(!memcmp(logit,oracle_logits[PROMPT],V*sizeof(float)));free(logit);
    assert(q36_mtp_kv_rows(reference,&oracle_hidden[PROMPT-1][0],oracle_ids+PROMPT,1,PROMPT-1));
    same_prefix(g_q36_mtp,reference,PROMPT);pending_matches(PROMPT+1);
    int length=q36_mtp_length(g_q36_mtp),capacity=q36_mtp_capacity(g_q36_mtp);
    assert(!q36_mtp_reserve(g_q36_mtp,-1));
    assert(q36_mtp_length(g_q36_mtp)==length&&q36_mtp_capacity(g_q36_mtp)==capacity);
    same_prefix(g_q36_mtp,reference,length);
    reset_recurrent(&m);assert(pin_restore(&m,oracle_ids,12)==0);
    assert(g_mp_len==0&&g_mp_pending_n==0&&!q36_mp_ready(PROMPT));
    q36_mtp_free(reference);close_active();target_free(&m);
}

int main(void) {
    setenv("COLI_LOOKUP","0",1);setenv("COLI_SPEC_GATE","0",1);
    setenv("COLI_DENSE_BITS","8",1);setenv("COLI_DENSE_I8","1",1);setenv("COLI_DENSE_IDOT","1",1);
    setenv("COLI_NO_OMP_TUNE","1",1);unsetenv("COLI_LOOKUP_FORCE");
#ifdef _OPENMP
    omp_set_dynamic(0);omp_set_num_threads(2);
#endif
    make_oracle();
    const int bits[]={8,32};
    for(size_t b=0;b<sizeof bits/sizeof bits[0];b++){
        prefill_equivalence(bits[b]);
        speculative_mode(bits[b],'a',0,23);speculative_mode(bits[b],'w',1,23);
        speculative_mode(bits[b],'w',2,23);speculative_mode(bits[b],'c',0,29);
        predicted_tail_is_replayed(bits[b]);cancellation_and_budget(bits[b]);pin_and_growth(bits[b]);
    }
    puts("MTP frontier: actual target logits, teacherforced KV/pending, predicted-tail replay, cancel/budgets and pin/growth exact");
    return 0;
}
