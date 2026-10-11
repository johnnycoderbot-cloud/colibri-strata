/* Authentic one-layer Qwen3.5 MTP cell. Include after qwen36's CPU helpers.
 * Pair (post-final-norm target h[p], E(token[p+1])) lives at cell/RoPE p
 * and predicts token[p+2]. Deeper drafts feed the previous NORMALIZED MTP
 * output, with their own causal KV tail. The caller must replace provisional
 * pairs with teacher-forced target pairs after verification.
 *
 * Source norms are raw Gemma RMSNorm weights (multiply by 1+w); FC concatenates
 * normalized embedding FIRST, normalized hidden second. Q width need not equal
 * hidden width. Source matrices use INT8 rows/INT8 activations in production;
 * the explicit 32-bit factory is an F32 reference, not an integer claim.
 * A cell owns weights, scratch and KV; target embedding/readout are borrowed.
 * Calls on one cell require external serialization. No target state is written.
 */
#ifndef COLI_QWEN36_MTP_H
#define COLI_QWEN36_MTP_H
#include "evidence_digest.h"
#include "strata_native_heads.h"
#include <limits.h>
#include <float.h>
#include <sys/stat.h>
#include <fcntl.h>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

#define Q36_MTP_KV_LIMIT ((size_t)512 << 20)
#define Q36_MTP_SCRATCH_LIMIT ((size_t)32 << 20)
#define Q36_MTP_WEIGHT_LIMIT ((size_t)512 << 20)
#define Q36_MTP_BATCH 32
#define Q36_MTP_PRECISION "qwen36:DBITS=8;IDOT=1;kernel=cpu;tap=post-final-norm-mtp/v1"

enum {
    Q36_MTP_FC, Q36_MTP_EMBED_NORM, Q36_MTP_HIDDEN_NORM, Q36_MTP_INPUT_NORM,
    Q36_MTP_Q, Q36_MTP_K, Q36_MTP_V, Q36_MTP_O, Q36_MTP_Q_NORM,
    Q36_MTP_K_NORM, Q36_MTP_POST_NORM, Q36_MTP_GATE, Q36_MTP_UP,
    Q36_MTP_DOWN, Q36_MTP_FINAL_NORM, Q36_MTP_TENSOR_COUNT
};
typedef struct {
    const float *values[Q36_MTP_TENSOR_COUNT];
    size_t counts[Q36_MTP_TENSOR_COUNT];
} Q36MtpWeights;
typedef struct {
    const Model *target;
    Model shadow;
    Layer layer;
    QW fc;
    float *embed_norm, *hidden_norm, *final_norm;
    float *keys, *values, *scores, *work;
    float *concat, *front, *input, *krows, *vrows, *qrows, *query, *context;
    float *att, *post, *gate, *up, *raw, *normal, *logits;
    int8_t *xq;
    float sx[Q36_MTP_BATCH];
    size_t work_count, xq_count, weight_bytes;
    int h, f, qwidth, kvwidth, bits, cap, len, threads;
#ifdef COLI_QWEN_MTP_TEST
    unsigned kv_projections, full_cells, lm_projections;
#endif
} Q36Mtp;

static const char *const q36_mtp_tensor_names[Q36_MTP_TENSOR_COUNT] = {
    "mtp.fc.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight", "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight", "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight", "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight", "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.mlp.gate_proj.weight", "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight", "mtp.norm.weight"
};

/* Bit inspection remains valid even when an embedding engine uses fast-math. */
static int q36_mtp_finite(const float *p,size_t n) {
    if(!p)return 0;
    for(size_t i=0;i<n;i++){uint32_t u;memcpy(&u,p+i,4);if((u&0x7f800000u)==0x7f800000u)return 0;}
    return 1;
}
static int q36_mtp_overlap(const void *a,size_t an,const void *b,size_t bn) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(!an||!bn)return 0;
    if(!a||!b||an>UINTPTR_MAX-x||bn>UINTPTR_MAX-y)return 1;
    return x<y+bn && y<x+an;
}
static int q36_mtp_geometry(const Model *t,int *f) {
    if(!t)return 0;
    const Cfg *c=&t->c;*f=c->inter>0?c->inter:c->shared_inter;
    return c->hidden>0 && c->hidden<=4096 && *f>0 && *f<=32768 &&
        c->vocab>0 && c->vocab<=1000000 &&
        (size_t)c->vocab<=SIZE_MAX/((size_t)c->hidden*sizeof(float)) &&
        c->q_heads>0 && c->q_heads<=128 &&
        c->kv_heads>0 && c->kv_heads<=c->q_heads && c->q_heads%c->kv_heads==0 &&
        c->head_dim>0 && c->head_dim<=512 && c->rotary_dim>0 &&
        c->rotary_dim<=c->head_dim && !(c->rotary_dim&1) &&
        q36_mtp_finite(&c->eps,1) && c->eps>0 &&
        q36_mtp_finite(&c->theta,1) && c->theta>0 &&
        (t->embed||t->embed_h) &&
        (t->lm_head.w||t->lm_head.q||t->lm_head.h) && !t->lm_head.q4 &&
        t->lm_head.I==c->hidden && t->lm_head.O==c->vocab &&
        (!t->lm_head.q||t->lm_head.sc);
}
static void q36_mtp_dims(const Cfg *c,int f,int rows[15],int cols[15]) {
    int h=c->hidden,hd=c->head_dim,q=c->q_heads*hd,k=c->kv_heads*hd;
    const int r[15]={h,h,h,h,2*q,k,k,h,hd,hd,h,f,f,h,h};
    const int z[15]={2*h,1,1,1,h,h,h,q,1,1,1,h,h,f,1};
    memcpy(rows,r,sizeof(r));memcpy(cols,z,sizeof(z));
}
static int q36_mtp_capacity(const Q36Mtp *m){return m?m->cap:0;}
static int q36_mtp_length(const Q36Mtp *m){return m?m->len:0;}
static int q36_mtp_truncate(Q36Mtp *m,int len) {
    if(!m||len<0||len>m->len)return 0;
    m->len=len;m->shadow.kv_len=len;return 1;
}
static void q36_mtp_free(Q36Mtp *m) {
    if(!m)return;
    qw_free(&m->fc);qw_free(&m->layer.q);qw_free(&m->layer.k);qw_free(&m->layer.v);
    qw_free(&m->layer.o);qw_free(&m->layer.sh_g);qw_free(&m->layer.sh_u);qw_free(&m->layer.sh_d);
    free(m->embed_norm);free(m->hidden_norm);free(m->final_norm);
    free(m->layer.in_ln);free(m->layer.post_ln);free(m->layer.qn);free(m->layer.kn);
    free(m->keys);free(m->values);free(m->scores);free(m->work);free(m->xq);free(m);
}
/* Growth preserves every valid per-head prefix despite the changed stride.
 * Allocate all replacements before publishing; never shrink or discard on error. */
static int q36_mtp_reserve(Q36Mtp *m,int cap) {
    if(!m||cap<1||cap>QWEN36_ATTN_MAX_CTX)return 0;
    if(cap<=m->cap)return 1;
    if((size_t)cap>SIZE_MAX/(size_t)m->kvwidth ||
       (size_t)cap>SIZE_MAX/(size_t)m->threads)return 0;
    size_t kv=(size_t)cap*m->kvwidth,sc=(size_t)cap*m->threads;
    if(kv>Q36_MTP_KV_LIMIT/(2*sizeof(float)) ||
       sc>Q36_MTP_SCRATCH_LIMIT/sizeof(float) ||
       (m->work_count+sc)*sizeof(float)+m->xq_count>Q36_MTP_SCRATCH_LIMIT)return 0;
    float *k=calloc(kv,sizeof(float)),*v=calloc(kv,sizeof(float)),*s=malloc(sc*sizeof(float));
    if(!k||!v||!s){free(k);free(v);free(s);return 0;}
    int hd=m->shadow.c.head_dim;
    for(int head=0;head<m->shadow.c.kv_heads;head++) {
        if(m->len) {
            memcpy(k+(size_t)head*cap*hd,m->keys+(size_t)head*m->cap*hd,(size_t)m->len*hd*sizeof(float));
            memcpy(v+(size_t)head*cap*hd,m->values+(size_t)head*m->cap*hd,(size_t)m->len*hd*sizeof(float));
        }
    }
    free(m->keys);free(m->values);free(m->scores);m->keys=k;m->values=v;m->scores=s;m->cap=cap;
    m->shadow.max_t=m->shadow.kv_cap=cap;m->shadow.attn_sc=s;return 1;
}
static int q36_mtp_copy_matrix(QW *dst,const float *src,int rows,int cols,int bits) {
    dst->I=cols;dst->O=rows;
    size_t n=(size_t)rows*cols;
    if(bits==32) {
        float *p=malloc(n*sizeof(float));if(!p)return 0;memcpy(p,src,n*sizeof(float));dst->w=p;return 1;
    }
    dst->q=q36_walloc(n);dst->sc=malloc((size_t)rows*sizeof(float));
    if(!dst->q||!dst->sc)return 0;
    #pragma omp parallel for schedule(static)
    for(int r=0;r<rows;r++) {
        const float *x=src+(size_t)r*cols;float am=0;
        for(int j=0;j<cols;j++){float a=fabsf(x[j]);if(a>am)am=a;}
        float scale=am>1e-12f?am/127.f:1.f;dst->sc[r]=scale;float inv=1.f/scale;
        for(int j=0;j<cols;j++){int q=(int)lrintf(x[j]*inv);if(q>127)q=127;if(q< -127)q=-127;dst->q[(size_t)r*cols+j]=(int8_t)q;}
    }
    return q36_mtp_finite(dst->sc,(size_t)rows);
}
/* The test factory is also the validated load implementation. It copies every
 * source tensor. The source is not scanned again during cache/predict calls. */
static Q36Mtp *q36_mtp_from_weights(const Model *target,const Q36MtpWeights *w,int cap,int bits) {
    int f,rows[15],cols[15];
    if(!w||cap<1||cap>QWEN36_ATTN_MAX_CTX||
       !q36_mtp_geometry(target,&f)||(bits!=8&&bits!=32))return NULL;
    q36_mtp_dims(&target->c,f,rows,cols);size_t total=0;
    for(int i=0;i<15;i++) {
        size_t n=(size_t)rows[i]*cols[i];
        if(w->counts[i]!=n||n>Q36_MTP_WEIGHT_LIMIT/sizeof(float)||
           total>Q36_MTP_WEIGHT_LIMIT-n*sizeof(float)||!q36_mtp_finite(w->values[i],n))return NULL;
        total+=n*sizeof(float);
    }
    Q36Mtp *m=calloc(1,sizeof(*m));if(!m)return NULL;
    m->target=target;m->h=target->c.hidden;m->f=f;m->bits=bits;m->weight_bytes=total;
    m->qwidth=target->c.q_heads*target->c.head_dim;m->kvwidth=target->c.kv_heads*target->c.head_dim;
    m->shadow.c=target->c;m->shadow.c.n_layers=1;m->shadow.c.q_head_dim=2*target->c.head_dim;
    m->shadow.c.k_head_dim=m->shadow.c.v_head_dim=target->c.head_dim;
    m->shadow.c.o_in=m->qwidth;m->shadow.c.attn_output_gate=1;
    m->shadow.L=&m->layer;m->threads=1;
#ifdef _OPENMP
    m->threads=omp_get_max_threads();if(m->threads>target->c.q_heads)m->threads=target->c.q_heads;
#endif
    /* Fixed batch scratch plus one full prediction and one vocabulary row. */
    size_t b=Q36_MTP_BATCH,h=(size_t)m->h,k=(size_t)m->kvwidth,q=(size_t)m->qwidth;
    m->work_count=b*(4*h+2*k)+4*q+5*h+2*(size_t)f+(size_t)target->c.vocab;
    int max_i=2*m->h;if(f>max_i)max_i=f;if(m->qwidth>max_i)max_i=m->qwidth;
    m->xq_count=b*(size_t)max_i;
    if(m->work_count>Q36_MTP_SCRATCH_LIMIT/sizeof(float) ||
       m->work_count*sizeof(float)+m->xq_count>Q36_MTP_SCRATCH_LIMIT){q36_mtp_free(m);return NULL;}
    m->work=malloc(m->work_count*sizeof(float));m->xq=malloc(m->xq_count);
    if(!m->work||!m->xq){q36_mtp_free(m);return NULL;}
    float *p=m->work;
    m->concat=p;p+=b*2*h;m->front=p;p+=b*h;m->input=p;p+=b*h;
    m->krows=p;p+=b*k;m->vrows=p;p+=b*k;m->qrows=p;p+=2*q;
    m->query=p;p+=q;m->context=p;p+=q;m->att=p;p+=h;m->post=p;p+=h;
    m->gate=p;p+=f;m->up=p;p+=f;m->raw=p;p+=h;m->normal=p;p+=h;m->logits=p;p+=target->c.vocab;
    /* Five h rows above includes one spare: this guard catches layout drift. */
    if((size_t)(p-m->work)>m->work_count){q36_mtp_free(m);return NULL;}
    QW *mat[15]={&m->fc,NULL,NULL,NULL,&m->layer.q,&m->layer.k,&m->layer.v,&m->layer.o,
        NULL,NULL,NULL,&m->layer.sh_g,&m->layer.sh_u,&m->layer.sh_d,NULL};
    float **norm[15]={NULL,&m->embed_norm,&m->hidden_norm,&m->layer.in_ln,NULL,NULL,NULL,NULL,
        &m->layer.qn,&m->layer.kn,&m->layer.post_ln,NULL,NULL,NULL,&m->final_norm};
    for(int i=0;i<15;i++) {
        if(mat[i]) {if(!q36_mtp_copy_matrix(mat[i],w->values[i],rows[i],cols[i],bits)){q36_mtp_free(m);return NULL;}}
        else {size_t n=(size_t)rows[i];*norm[i]=malloc(n*sizeof(float));
            if(!*norm[i]){q36_mtp_free(m);return NULL;}memcpy(*norm[i],w->values[i],n*sizeof(float));}
    }
    if(!q36_mtp_reserve(m,cap)){q36_mtp_free(m);return NULL;}return m;
}

/* CPU-only dispatch. Quantized source matrices ALWAYS use the same native
 * INT8 activation rounding and dot scale order. A catch-up K/V call uses this
 * identical path, without computing unused Q, MLP or vocabulary projections. */
static int q36_mtp_linear(Q36Mtp *m,const QW *w,float *out,const float *in,int S) {
    if(!q36_mtp_finite(in,(size_t)S*w->I))return 0;
    if(w->q) {
        for(int s=0;s<S;s++)m->sx[s]=dense_act_i8(in+(size_t)s*w->I,w->I,m->xq+(size_t)s*w->I,NULL);
        matmul_q_idot(out,m->xq,m->sx,w->q,w->sc,S,w->I,w->O);
    } else if(w->h)matmul_h(out,in,w->h,S,w->I,w->O);
    else matmul(out,in,w->w,S,w->I,w->O);
    return q36_mtp_finite(out,(size_t)S*w->O);
}
static int q36_mtp_borrowed_overlap(const QW *w,const void *p,size_t n) {
    size_t count=(size_t)w->I*w->O;
    return (w->w&&q36_mtp_overlap(p,n,w->w,count*4)) ||
        (w->q&&q36_mtp_overlap(p,n,w->q,count)) ||
        (w->sc&&q36_mtp_overlap(p,n,w->sc,(size_t)w->O*4)) ||
        (w->h&&q36_mtp_overlap(p,n,w->h,count*2));
}
static int q36_mtp_internal_overlap(const Q36Mtp *m,const void *p,size_t n) {
    const QW *mat[8]={&m->fc,&m->layer.q,&m->layer.k,&m->layer.v,&m->layer.o,
        &m->layer.sh_g,&m->layer.sh_u,&m->layer.sh_d};
    const float *norm[7]={m->embed_norm,m->hidden_norm,m->layer.in_ln,m->layer.qn,m->layer.kn,m->layer.post_ln,m->final_norm};
    for(int i=0;i<8;i++)if(q36_mtp_borrowed_overlap(mat[i],p,n))return 1;
    for(int i=0;i<7;i++)if(q36_mtp_overlap(p,n,norm[i],(size_t)((i==3||i==4)?m->shadow.c.head_dim:m->h)*4))return 1;
    size_t emb=(size_t)m->target->c.vocab*m->h;
    return q36_mtp_overlap(p,n,m,sizeof(*m)) ||
        q36_mtp_overlap(p,n,m->work,m->work_count*4) ||
        q36_mtp_overlap(p,n,m->xq,m->xq_count) ||
        q36_mtp_overlap(p,n,m->keys,(size_t)m->cap*m->kvwidth*4) ||
        q36_mtp_overlap(p,n,m->values,(size_t)m->cap*m->kvwidth*4) ||
        q36_mtp_overlap(p,n,m->scores,(size_t)m->cap*m->threads*4) ||
        (m->target->embed&&q36_mtp_overlap(p,n,m->target->embed,emb*4)) ||
        (m->target->embed_h&&q36_mtp_overlap(p,n,m->target->embed_h,emb*2)) ||
        q36_mtp_borrowed_overlap(&m->target->lm_head,p,n);
}
static int q36_mtp_prepare(Q36Mtp *m,const float *h,const int *tokens,int S,int pos) {
    int d=m->h,hd=m->shadow.c.head_dim,kv=m->shadow.c.kv_heads;
    for(int s=0;s<S;s++) {
        float *cat=m->concat+(size_t)s*2*d;int id=tokens[s];
        if(m->target->embed_h)f16_to_f32_bulk(m->target->embed_h+(size_t)id*d,cat,d);
        else memcpy(cat,m->target->embed+(size_t)id*d,(size_t)d*4);
        if(!q36_mtp_finite(cat,(size_t)d))return 0;
        rmsnorm_row(cat,cat,m->embed_norm,d,m->shadow.c.eps);
        rmsnorm_row(cat+d,h+(size_t)s*d,m->hidden_norm,d,m->shadow.c.eps);
    }
    if(!q36_mtp_linear(m,&m->fc,m->front,m->concat,S))return 0;
    for(int s=0;s<S;s++)rmsnorm_row(m->input+(size_t)s*d,m->front+(size_t)s*d,m->layer.in_ln,d,m->shadow.c.eps);
    if(!q36_mtp_linear(m,&m->layer.k,m->krows,m->input,S) ||
       !q36_mtp_linear(m,&m->layer.v,m->vrows,m->input,S))return 0;
    for(int s=0;s<S;s++)for(int j=0;j<kv;j++) {
        float *k=m->krows+((size_t)s*kv+j)*hd;
        rmsnorm_row(k,k,m->layer.kn,hd,m->shadow.c.eps);
        rope_head_partial(k,pos+s,m->shadow.c.rotary_dim,hd,m->shadow.c.theta);
    }
    return q36_mtp_finite(m->krows,(size_t)S*m->kvwidth);
}
static void q36_mtp_store(Q36Mtp *m,int S,int pos) {
    int hd=m->shadow.c.head_dim,kv=m->shadow.c.kv_heads;
    for(int s=0;s<S;s++)for(int j=0;j<kv;j++) {
        memcpy(m->keys+((size_t)j*m->cap+pos+s)*hd,m->krows+((size_t)s*kv+j)*hd,(size_t)hd*4);
        memcpy(m->values+((size_t)j*m->cap+pos+s)*hd,m->vrows+((size_t)s*kv+j)*hd,(size_t)hd*4);
    }
}
static int q36_mtp_input_ok(const Q36Mtp *m,const float *h,const int *tokens,int S,int pos) {
    if(!m||!h||!tokens||S<1||S>Q36_MTP_BATCH||pos!=m->len||pos<0||S>m->cap-pos ||
       q36_mtp_internal_overlap(m,h,(size_t)S*m->h*4) ||
       q36_mtp_internal_overlap(m,tokens,(size_t)S*sizeof(int)) ||
       !q36_mtp_finite(h,(size_t)S*m->h))return 0;
    for(int i=0;i<S;i++)if(tokens[i]<0||tokens[i]>=m->target->c.vocab)return 0;
    return 1;
}
static int q36_mtp_kv_rows(Q36Mtp *m,const float *hidden,const int *next,int S,int pos) {
    if(!q36_mtp_input_ok(m,hidden,next,S,pos)||!q36_mtp_prepare(m,hidden,next,S,pos))return 0;
    q36_mtp_store(m,S,pos);m->len=pos+S;m->shadow.kv_len=m->len;
#ifdef COLI_QWEN_MTP_TEST
    m->kv_projections+=(unsigned)S;
#endif
    return 1;
}
static int q36_mtp_predict_one(Q36Mtp *m,const float *hidden,int next,int pos,float *out_hidden,float *raw_logits) {
    if(!q36_mtp_input_ok(m,hidden,&next,1,pos)||!out_hidden||!raw_logits ||
       q36_mtp_overlap(out_hidden,(size_t)m->h*4,hidden,(size_t)m->h*4) ||
       q36_mtp_overlap(raw_logits,(size_t)m->target->c.vocab*4,hidden,(size_t)m->h*4) ||
       q36_mtp_overlap(out_hidden,(size_t)m->h*4,raw_logits,(size_t)m->target->c.vocab*4) ||
       q36_mtp_internal_overlap(m,out_hidden,(size_t)m->h*4) ||
       q36_mtp_internal_overlap(m,raw_logits,(size_t)m->target->c.vocab*4) ||
       !q36_mtp_prepare(m,hidden,&next,1,pos) ||
       !q36_mtp_linear(m,&m->layer.q,m->qrows,m->input,1))return 0;
    int hd=m->shadow.c.head_dim,qh=m->shadow.c.q_heads,group=qh/m->shadow.c.kv_heads;
    for(int j=0;j<qh;j++) {
        float *query=m->query+(size_t)j*hd;
        memcpy(query,m->qrows+(size_t)j*2*hd,(size_t)hd*4);
        rmsnorm_row(query,query,m->layer.qn,hd,m->shadow.c.eps);
        rope_head_partial(query,pos,m->shadow.c.rotary_dim,hd,m->shadow.c.theta);
    }
    if(!q36_mtp_finite(m->query,(size_t)m->qwidth))return 0;
    q36_mtp_store(m,1,pos); /* A failed tail is outside the unchanged valid prefix. */
    float scale=1.f/sqrtf((float)hd);
    #pragma omp parallel for schedule(static) num_threads(m->threads)
    for(int j=0;j<qh;j++) {
        int tid=0;
#ifdef _OPENMP
        tid=omp_get_thread_num();
#endif
        int kv=j/group;float *score=m->scores+(size_t)tid*m->cap;
        float *ctx=m->context+(size_t)j*hd;const float *q=m->query+(size_t)j*hd;
        for(int t=0;t<=pos;t++)score[t]=dot_f32_lanes(q,m->keys+((size_t)kv*m->cap+t)*hd,hd)*scale;
        softmax_row(score,pos+1);
        for(int d=0;d<hd;d++)ctx[d]=0;
        for(int t=0;t<=pos;t++) {
            const float *v=m->values+((size_t)kv*m->cap+t)*hd;
            for(int d=0;d<hd;d++)ctx[d]+=score[t]*v[d];
        }
        for(int d=0;d<hd;d++)ctx[d]*=1.f/(1.f+expf(-m->qrows[(size_t)j*2*hd+hd+d]));
    }
    if(!q36_mtp_linear(m,&m->layer.o,m->att,m->context,1))return 0;
    for(int d=0;d<m->h;d++)m->raw[d]=m->front[d]+m->att[d];
    if(!q36_mtp_finite(m->raw,(size_t)m->h))return 0;
    rmsnorm_row(m->post,m->raw,m->layer.post_ln,m->h,m->shadow.c.eps);
    if(!q36_mtp_linear(m,&m->layer.sh_g,m->gate,m->post,1) ||
       !q36_mtp_linear(m,&m->layer.sh_u,m->up,m->post,1))return 0;
    for(int i=0;i<m->f;i++){float g=m->gate[i];m->gate[i]=(g/(1.f+expf(-g)))*m->up[i];}
    if(!q36_mtp_linear(m,&m->layer.sh_d,m->att,m->gate,1))return 0;
    for(int d=0;d<m->h;d++)m->raw[d]+=m->att[d];
    if(!q36_mtp_finite(m->raw,(size_t)m->h))return 0;
    rmsnorm_row(m->normal,m->raw,m->final_norm,m->h,m->shadow.c.eps);
    if(!q36_mtp_linear(m,&m->target->lm_head,m->logits,m->normal,1))return 0;
    memcpy(out_hidden,m->normal,(size_t)m->h*4);memcpy(raw_logits,m->logits,(size_t)m->target->c.vocab*4);
    m->len=pos+1;m->shadow.kv_len=m->len;
#ifdef COLI_QWEN_MTP_TEST
    m->full_cells++;m->lm_projections++;
#endif
    return 1;
}
#ifdef COLI_QWEN_MTP_TEST
/* Explicitly permits provisional allocated tails; not a valid-prefix getter. */
static int q36_mtp_test_peek_allocated(const Q36Mtp *m,int pos,float *k,float *v) {
    if(!m||!k||!v||pos<0||pos>=m->cap)return 0;
    int hd=m->shadow.c.head_dim;
    for(int j=0;j<m->shadow.c.kv_heads;j++) {
        memcpy(k+(size_t)j*hd,m->keys+((size_t)j*m->cap+pos)*hd,(size_t)hd*4);
        memcpy(v+(size_t)j*hd,m->values+((size_t)j*m->cap+pos)*hd,(size_t)hd*4);
    }
    return 1;
}
#endif

/* Fail-closed sidecar loader: fixed filenames, regular files, bounded reads,
 * exact payload digest, unique JSON keys and exact raw-F32 tensor geometry. */
static unsigned char *q36_mtp_read(const char *dir,const char *name,size_t limit,size_t *n) {
    if(!dir||strlen(dir)>4096)return NULL;
    size_t size=strlen(dir)+strlen(name)+2;char *path=malloc(size);if(!path)return NULL;
    snprintf(path,size,"%s/%s",dir,name);FILE *f=NULL;
#ifdef _WIN32
    int wn=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path,-1,NULL,0);wchar_t *wide=wn?malloc((size_t)wn*sizeof(*wide)):NULL;
    if(wide&&MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path,-1,wide,wn)) {
        HANDLE handle=CreateFileW(wide,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
        if(handle!=INVALID_HANDLE_VALUE) {
            if(GetFileType(handle)==FILE_TYPE_DISK) {
                int fd=_open_osfhandle((intptr_t)handle,_O_RDONLY|_O_BINARY);
                if(fd>=0){f=_fdopen(fd,"rb");if(!f)_close(fd);}else CloseHandle(handle);
            }else CloseHandle(handle);
        }
    }
    free(wide);
#else
    int fd=open(path,O_RDONLY|O_NONBLOCK);struct stat st;
    if(fd>=0){if(!fstat(fd,&st)&&S_ISREG(st.st_mode))f=fdopen(fd,"rb");if(!f)close(fd);}
#endif
    free(path);if(!f)return NULL;
    long length;if(fseek(f,0,SEEK_END)||(length=ftell(f))<1||(size_t)length>limit||fseek(f,0,SEEK_SET)){fclose(f);return NULL;}
    unsigned char *data=malloc((size_t)length+1);
    if(!data||fread(data,1,(size_t)length,f)!=(size_t)length||fgetc(f)!=EOF||ferror(f)) {free(data);fclose(f);return NULL;}
    fclose(f);data[length]=0;*n=(size_t)length;return data;
}
static int q36_mtp_unique(const jval *v,int depth) {
    if(!v||depth>16)return 0;
    if(v->t==J_OBJ)for(int i=0;i<v->len;i++)for(int j=i+1;j<v->len;j++)if(!strcmp(v->keys[i],v->keys[j]))return 0;
    if(v->t==J_OBJ||v->t==J_ARR)for(int i=0;i<v->len;i++)if(!q36_mtp_unique(v->kids[i],depth+1))return 0;
    return 1;
}
static int q36_mtp_string(const jval *o,const char *key,const char *s) {
    jval *v=json_get((jval*)o,key);return v&&v->t==J_STR&&!strcmp(v->str,s);
}
static int q36_mtp_number(const jval *o,const char *key,double n) {
    jval *v=json_get((jval*)o,key);return v&&v->t==J_NUM&&v->num==n;
}
static int q36_mtp_true(const jval *o,const char *key) {
    jval *v=json_get((jval*)o,key);return v&&v->t==J_BOOL&&v->boolean;
}
static int q36_mtp_shape(const jval *t,int rows,int cols,int vector) {
    jval *s=json_get((jval*)t,"shape");
    return s&&s->t==J_ARR&&s->len==(vector?1:2)&&s->kids[0]->t==J_NUM&&s->kids[0]->num==rows&&
        (vector||(s->kids[1]->t==J_NUM&&s->kids[1]->num==cols));
}
static uint32_t q36_mtp_u32(const unsigned char *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static uint64_t q36_mtp_u64(const unsigned char *p){return (uint64_t)q36_mtp_u32(p)|(uint64_t)q36_mtp_u32(p+4)<<32;}
static Q36Mtp *q36_mtp_init(const Model *target,const char *target_dir,const char *side_dir,int cap) {
    size_t mn=0,bn=0;int f,rows[15],cols[15];Q36Mtp *result=NULL;
    if(!q36_mtp_geometry(target,&f)||cap<1||cap>QWEN36_ATTN_MAX_CTX)return NULL;
    unsigned char *meta=q36_mtp_read(side_dir,"qwen36_mtp.json",128*1024,&mn),*blob=NULL;
    jval *root=NULL,*header=NULL;Q36MtpWeights weights={{0},{0}};float *copies[15]={0};
    if(!meta||memchr(meta,0,mn)||strstr((char*)meta,"\\u0000"))goto done;
    root=json_parse_checked((char*)meta);if(!root||root->t!=J_OBJ||!q36_mtp_unique(root,0))goto done;
    if(!q36_mtp_string(root,"schema","colibri.qwen-mtp/v1")||!q36_mtp_string(root,"tensor_dtype","F32")||
       !q36_mtp_string(root,"norm_convention","qwen_rmsnorm_plus_one")||
       !q36_mtp_string(root,"target_hidden_tap","post_final_norm")||
       !q36_mtp_true(root,"shared_embedding")||!q36_mtp_true(root,"shared_lm_head"))goto done;
    jval *order=json_get(root,"fc_order"),*geo=json_get(root,"geometry"),*native=json_get(root,"native");
    if(!order||order->t!=J_ARR||order->len!=2||order->kids[0]->t!=J_STR||order->kids[1]->t!=J_STR||
       strcmp(order->kids[0]->str,"embedding")||strcmp(order->kids[1]->str,"target_hidden"))goto done;
    const Cfg *c=&target->c;
    if(!q36_mtp_number(geo,"hidden_size",c->hidden)||!q36_mtp_number(geo,"intermediate_size",f)||
       !q36_mtp_number(geo,"num_attention_heads",c->q_heads)||!q36_mtp_number(geo,"num_key_value_heads",c->kv_heads)||
       !q36_mtp_number(geo,"head_dim",c->head_dim)||!q36_mtp_number(geo,"vocab_size",c->vocab)||
       !q36_mtp_number(geo,"num_hidden_layers",c->n_layers)||
       !q36_mtp_number(geo,"mtp_num_hidden_layers",1)||!q36_mtp_number(geo,"rotary_dim",c->rotary_dim)||
       !q36_mtp_number(geo,"rope_theta",c->theta)||!q36_mtp_true(geo,"attn_output_gate"))goto done;
    jval *epsilon=json_get(geo,"rms_norm_eps"),*partial=json_get(geo,"partial_rotary_factor");
    if(!epsilon||epsilon->t!=J_NUM||(float)epsilon->num!=c->eps||
       !partial||partial->t!=J_NUM||(float)partial->num!=c->partial_rotary_factor)goto done;
    uint8_t fp[32];char expected[65];const char hex[]="0123456789abcdef";
    if(!q36_mtp_string(native,"runtime_precision",Q36_MTP_PRECISION)||
       sn_fingerprint_model(target_dir,Q36_MTP_PRECISION,fp)!=SN_OK)goto done;
    for(int i=0;i<32;i++){expected[2*i]=hex[fp[i]>>4];expected[2*i+1]=hex[fp[i]&15];}expected[64]=0;
    if(!q36_mtp_string(native,"fingerprint",expected))goto done;
    jval *side=json_get(root,"sidecar"),*td=json_get(root,"tensors");
    if(!q36_mtp_string(side,"file","model-mtp.safetensors")||!td||td->t!=J_OBJ||td->len!=15)goto done;
    blob=q36_mtp_read(side_dir,"model-mtp.safetensors",Q36_MTP_WEIGHT_LIMIT,&bn);
    if(!blob||bn<10||!q36_mtp_number(side,"bytes",(double)bn))goto done;
    evidence_sha256_hex(blob,bn,expected);if(!q36_mtp_string(side,"sha256",expected))goto done;
    uint64_t hn=q36_mtp_u64(blob);
    if(hn<2||hn>128*1024||hn>bn-8||memchr(blob+8,0,(size_t)hn))goto done;
    unsigned char save=blob[8+(size_t)hn];blob[8+(size_t)hn]=0;
    if(strstr((char*)blob+8,"\\u0000")){blob[8+(size_t)hn]=save;goto done;}
    header=json_parse_checked((char*)blob+8);blob[8+(size_t)hn]=save;
    if(!header||header->t!=J_OBJ||!q36_mtp_unique(header,0)||header->len<15||header->len>16)goto done;
    if(header->len==16&&!json_get(header,"__metadata__"))goto done;
    q36_mtp_dims(c,f,rows,cols);size_t starts[15],ends[15],payload=bn-8-(size_t)hn;
    for(int i=0;i<15;i++) {
        jval *desc=json_get(header,q36_mtp_tensor_names[i]),*decl=json_get(td,q36_mtp_tensor_names[i]);
        int vector=cols[i]==1;
        if(!q36_mtp_string(desc,"dtype","F32")||!q36_mtp_shape(desc,rows[i],cols[i],vector)||
           !q36_mtp_string(decl,"dtype","F32")||!q36_mtp_shape(decl,rows[i],cols[i],vector))goto done;
        jval *off=json_get(desc,"data_offsets");size_t count=(size_t)rows[i]*cols[i];
        if(!off||off->t!=J_ARR||off->len!=2||off->kids[0]->t!=J_NUM||off->kids[1]->t!=J_NUM||
           off->kids[0]->num<0||off->kids[1]->num<0||off->kids[0]->num>payload||off->kids[1]->num>payload||
           off->kids[0]->num!=floor(off->kids[0]->num)||off->kids[1]->num!=floor(off->kids[1]->num))goto done;
        starts[i]=(size_t)off->kids[0]->num;ends[i]=(size_t)off->kids[1]->num;
        if(ends[i]<starts[i]||ends[i]-starts[i]!=count*4)goto done;
        copies[i]=malloc(count*4);if(!copies[i])goto done;
        const unsigned char *src=blob+8+(size_t)hn+starts[i];
        for(size_t j=0;j<count;j++){uint32_t u=q36_mtp_u32(src+4*j);memcpy(copies[i]+j,&u,4);}
        weights.values[i]=copies[i];weights.counts[i]=count;
    }
    /* Require a complete non-overlapping payload with no gaps/trailing bytes. */
    size_t position=0;for(int n=0;n<15;n++) {
        int found=-1;for(int i=0;i<15;i++)if(starts[i]==position){if(found>=0)goto done;found=i;}
        if(found<0)goto done;position=ends[found];
    }
    if(position!=payload)goto done;
    result=q36_mtp_from_weights(target,&weights,cap,8);
done:
    for(int i=0;i<15;i++)free(copies[i]);json_free(header);json_free(root);free(blob);free(meta);return result;
}
#endif
