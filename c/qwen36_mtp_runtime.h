/* Target-frontier bookkeeping for pretrained sequential Qwen MTP.
 * The cell owns weights/KV; this adapter owns confirmed target hidden rows.
 * Accepted tokens never make predicted MTP hidden/KV rows canonical. */
#ifndef COLI_QWEN36_MTP_RUNTIME_H
#define COLI_QWEN36_MTP_RUNTIME_H
#include "qwen36_mtp.h"
static Q36Mtp *g_q36_mtp;
static int g_mp_len, g_mp_pending_n, g_mp_next[Q36_SPEC_ROWS];
static int g_mp_h, g_mp_capture_rows, g_mp_depth;
static float *g_mp_pending, *g_mp_rows, *g_mp_chunk;
static float *g_mp_chain, *g_mp_logit;
static int *g_mp_oracle, g_mp_oracle_n, g_mp_oracle_prompt;
static int g_mp_force, g_mp_force_row;

typedef struct {
    int len, pending_n, next[Q36_SPEC_ROWS];
    float *pending;
} Q36MtpSaved;
static void q36_mp_saved_free(void *value) {
    Q36MtpSaved *s=value; if(!s)return;free(s->pending);free(s);
}
static void q36_mp_shutdown(void) {
    q36_mtp_free(g_q36_mtp);g_q36_mtp=NULL;
    free(g_mp_pending);free(g_mp_rows);free(g_mp_chunk);free(g_mp_chain);free(g_mp_logit);free(g_mp_oracle);
    g_mp_pending=g_mp_rows=g_mp_chunk=NULL;g_mp_oracle=NULL;
    g_mp_chain=g_mp_logit=NULL;
}
static void q36_mp_reset(void) {
    if(g_q36_mtp)q36_mtp_truncate(g_q36_mtp,0);
    g_mp_len=g_mp_pending_n=0;g_mp_capture_rows=0;
}
static int q36_mp_ready(int pos) {
    return g_q36_mtp && g_mp_pending_n>0 && g_mp_len+g_mp_pending_n==pos &&
           q36_mtp_length(g_q36_mtp)==g_mp_len;
}
/* Complete real target pairs, optionally projecting only the final pair. */
static int q36_mp_take(int token,float *hidden,float *logit) {
    int n=g_mp_pending_n; if(!g_q36_mtp || n<1)return 0;
    int next[Q36_SPEC_ROWS];
    for(int j=0;j+1<n;j++)next[j]=g_mp_next[j];next[n-1]=token;
    int earlier=logit ? n-1 : n;
    if(earlier && !q36_mtp_kv_rows(g_q36_mtp,g_mp_pending,next,earlier,g_mp_len))return 0;
    if(logit && !q36_mtp_predict_one(g_q36_mtp,g_mp_pending+(size_t)(n-1)*g_mp_h,
                                   token,g_mp_len+n-1,hidden,logit)) {
        q36_mtp_truncate(g_q36_mtp,g_mp_len);return 0;
    }
    g_mp_len+=n;g_mp_pending_n=0;return 1;
}
static void q36_mp_observe(const float *hidden,const int *ids,int S,int pos) {
    if(!g_q36_mtp || S<1)return;
    if(g_mp_len+g_mp_pending_n>pos) {
        g_mp_pending_n=0;
        if(g_mp_len>pos-1)g_mp_len=pos>0?pos-1:0;
        q36_mtp_truncate(g_q36_mtp,g_mp_len);
    }
    if(q36_mp_ready(pos) && !q36_mp_take(ids[0],NULL,NULL)) {
        q36_mp_reset();return;
    }
    if(g_mp_len!=pos){g_mp_pending_n=0;return;}
    if(S>1 && !q36_mtp_kv_rows(g_q36_mtp,hidden,ids+1,S-1,pos)) {
        q36_mp_reset();return;
    }
    g_mp_len=pos+S-1;
    memcpy(g_mp_pending,hidden+(size_t)(S-1)*g_mp_h,(size_t)g_mp_h*sizeof(float));
    g_mp_pending_n=1;
}
static void q36_mp_capture(Model *m,const float *x,const int *ids,int S,int pos) {
    if(!g_q36_mtp || S<1)return;
    if(m->vis_rows_n){q36_mp_reset();return;}
    if(g_mp_capture_rows==S && S<=Q36_SPEC_ROWS) {
        for(int r=0;r<S;r++)rmsnorm_row(g_mp_rows+(size_t)r*g_mp_h,
            x+(size_t)r*g_mp_h,m->final_norm,g_mp_h,m->c.eps);
        return;
    }
    for(int base=0;base<S;) {
        int rows=S-base<32?S-base:32;
        for(int r=0;r<rows;r++)rmsnorm_row(g_mp_chunk+(size_t)r*g_mp_h,
            x+(size_t)(base+r)*g_mp_h,m->final_norm,g_mp_h,m->c.eps);
        q36_mp_observe(g_mp_chunk,ids+base,rows,pos+base);base+=rows;
    }
}
static void q36_mp_settle(const float *hidden,const int *ids,int pos,int keep) {
    if(!g_q36_mtp || !hidden || keep<1 || keep>Q36_SPEC_ROWS)return;
    /* Provisional deeper draft rows are overwritten by these real pairs. */
    if(!q36_mtp_truncate(g_q36_mtp,pos)){q36_mp_reset();return;}
    g_mp_len=pos;g_mp_pending_n=keep;
    memcpy(g_mp_pending,hidden,(size_t)keep*g_mp_h*sizeof(float));
    for(int j=0;j+1<keep;j++)g_mp_next[j]=ids[j+1];
}
static Q36MtpSaved *q36_mp_save(Q36MtpSaved *s) {
    if(!g_q36_mtp)return s;
    if(!s) {
        s=calloc(1,sizeof(*s));if(!s)return NULL;
        s->pending=malloc((size_t)Q36_SPEC_ROWS*g_mp_h*sizeof(float));
        if(!s->pending){free(s);return NULL;}
    }
    s->len=g_mp_len;s->pending_n=g_mp_pending_n;
    memcpy(s->next,g_mp_next,sizeof(s->next));
    memcpy(s->pending,g_mp_pending,(size_t)g_mp_pending_n*g_mp_h*sizeof(float));
    return s;
}
static int q36_mp_restore(const Q36MtpSaved *s) {
    if(!g_q36_mtp)return 1;
    if(!s || s->pending_n<0 || s->pending_n>Q36_SPEC_ROWS ||
       !q36_mtp_truncate(g_q36_mtp,s->len)){q36_mp_reset();return 0;}
    g_mp_len=s->len;g_mp_pending_n=s->pending_n;
    memcpy(g_mp_next,s->next,sizeof(g_mp_next));
    memcpy(g_mp_pending,s->pending,(size_t)s->pending_n*g_mp_h*sizeof(float));
    return 1;
}
static void q36_mp_test_config(Model *m) {
    const char *mode=getenv("Q36_MTP_FORCE"),*path=getenv("Q36_MTP_TEST_ORACLE");
    if(!mode || !*mode) {
        if(path && *path){fprintf(stderr,"Q36_MTP_TEST_ORACLE requires Q36_MTP_FORCE\n");exit(1);}
        return;
    }
    if(!strcmp(mode,"accept"))g_mp_force='a';
    else if(!strcmp(mode,"cycle"))g_mp_force='c';
    else if(!strcmp(mode,"row1")){g_mp_force='w';g_mp_force_row=1;}
    else if(!strcmp(mode,"row2")){g_mp_force='w';g_mp_force_row=2;}
    else {fprintf(stderr,"Q36_MTP_FORCE must be accept, cycle, row1 or row2\n");exit(1);}
    if(!path || !*path)return; /* CLI reference oracle, if provided later. */
    size_t path_len=strlen(path);char *directory=malloc(path_len+2);
    if(path_len>4096 || !directory){free(directory);fprintf(stderr,"[qwen36 MTP] invalid test oracle path\n");exit(1);}
    memcpy(directory,path,path_len+1);char *slash=strrchr(directory,'/'),*back=strrchr(directory,'\\');
    if(back && (!slash || back>slash))slash=back;
    const char *name=path;const char *dir=".";
    if(slash){size_t split=(size_t)(slash-directory);name=path+split+1;
        if(!split)directory[1]=0;else *slash=0;dir=directory;}
    size_t bytes=0;unsigned char *text=q36_mtp_read(dir,name,65536,&bytes);free(directory);
    if(!text){fprintf(stderr,"[qwen36 MTP] invalid regular test oracle file\n");exit(1);}
    if(memchr(text,0,bytes) || strstr((char*)text,"\\u0000")) {
        free(text);fprintf(stderr,"[qwen36 MTP] NUL in test oracle\n");exit(1);
    }
    jval *root=json_parse_checked((char*)text);free(text);
    jval *np=json_get(root,"prompt_tokens"),*tokens=json_get(root,"tokens");
    if(!root || !q36_mtp_unique(root,0) || root->t!=J_OBJ || !np || np->t!=J_NUM || np->num!=floor(np->num) ||
       np->num<1 || np->num>INT_MAX-512 || !tokens || tokens->t!=J_ARR ||
       tokens->len<1 || tokens->len>512) {
        json_free(root);fprintf(stderr,"[qwen36 MTP] invalid test oracle schema\n");exit(1);
    }
    g_mp_oracle=malloc((size_t)tokens->len*sizeof(int));
    if(!g_mp_oracle){json_free(root);fprintf(stderr,"[qwen36 MTP] test oracle allocation failed\n");exit(1);}
    for(int i=0;i<tokens->len;i++) {
        jval *v=tokens->kids[i];
        if(!v || v->t!=J_NUM || v->num!=floor(v->num) || v->num<0 || v->num>=m->c.vocab) {
            json_free(root);fprintf(stderr,"[qwen36 MTP] invalid test oracle token\n");exit(1);
        }
        g_mp_oracle[i]=(int)v->num;
    }
    g_mp_oracle_n=tokens->len;g_mp_oracle_prompt=(int)np->num;json_free(root);
    fprintf(stderr,"[qwen36 MTP] SYNTHETIC proposal overrides enabled for correctness testing only\n");
}
static void q36_mp_init(Model *m,const char *target_dir) {
    const char *path=getenv("Q36_MTP");if(!path || !*path || !strcmp(path,"0"))return;
    if(!dense_i8_on() || dense_bits()!=8 || !dense_idot_on() || qt_ready() ||
       g_cache_route || qq_active() || m->dn_dev || g_q36_mux_slots>1 ||
       q36_has_clef_head(target_dir) || g_sn_bank || m->c.n_experts) {
        fprintf(stderr,"[qwen36 MTP] requires qualified dense CPU INT8/IDOT text, one KV slot, without another head adapter\n");exit(1);
    }
#ifdef COLI_VULKAN
    if(g_vk_ready){fprintf(stderr,"[qwen36 MTP] active Vulkan cell qualification pending\n");exit(1);}
#endif
    const char *depth=getenv("Q36_MTP_DRAFTS");char *end=NULL;errno=0;
    long requested=depth&&*depth?strtol(depth,&end,10):3;
    if(errno || (depth&&*depth&&(end==depth||*end)) || requested<1 || requested>3) {
        fprintf(stderr,"Q36_MTP_DRAFTS must be 1, 2 or 3\n");exit(1);
    }
    g_mp_depth=(int)requested;
    /* The cell loader binds its sidecar to the actual target directory. */
    g_q36_mtp=q36_mtp_init(m,target_dir,path,m->kv_cap>0?m->kv_cap:1);
    if(!g_q36_mtp){fprintf(stderr,"[qwen36 MTP] sidecar refused\n");exit(1);}
    g_mp_h=m->c.hidden;
    g_mp_pending=malloc((size_t)Q36_SPEC_ROWS*g_mp_h*sizeof(float));
    g_mp_rows=malloc((size_t)Q36_SPEC_ROWS*g_mp_h*sizeof(float));
    g_mp_chunk=malloc((size_t)32*g_mp_h*sizeof(float));
    g_mp_chain=malloc((size_t)2*g_mp_h*sizeof(float));
    g_mp_logit=malloc((size_t)m->c.vocab*sizeof(float));
    if(!g_mp_pending || !g_mp_rows || !g_mp_chunk || !g_mp_chain || !g_mp_logit) {
        q36_mp_shutdown();fprintf(stderr,"[qwen36 MTP] frontier allocation failed\n");exit(1);
    }
    q36_mp_test_config(m);atexit(q36_mp_shutdown);
    fprintf(stderr,"[qwen36 MTP] pretrained sequential drafter ready, maximum %d drafts\n",g_mp_depth);
}
#endif
