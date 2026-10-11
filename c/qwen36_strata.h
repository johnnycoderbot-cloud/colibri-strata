/* CPU head ownership and frontier capture. The existing verifier owns rollback. */
#ifndef COLI_QWEN36_STRATA_H
#define COLI_QWEN36_STRATA_H
#include "strata_native_heads.h"
static sn_bank *g_sn_bank;
static sn_info g_sn_info;
static float *g_sn_frontier, *g_sn_rows, *g_sn_features;
/* Residual banks borrow the already-computed, unfiltered target distribution.
 * Own one copy: samplers may mutate returned logits after step_ex publishes. */
static float *g_sn_base;
static int g_sn_residual, g_sn_base_valid, g_sn_base_pos = -1;
static int g_sn_h, g_sn_valid, g_sn_pos = -1, g_sn_capture_rows;
static FILE *g_sn_trace;
static unsigned long long g_sn_generation;
static int g_sn_trace_active;
static int g_sn_embedding;
static int g_sn_frontier_token;
static void q36_sn_shutdown(void) {
    if (g_sn_trace) { fclose(g_sn_trace); g_sn_trace = NULL; }
    sn_close(g_sn_bank); g_sn_bank = NULL;
    free(g_sn_frontier); free(g_sn_rows); free(g_sn_features); g_sn_frontier = g_sn_rows = g_sn_features = NULL;
    free(g_sn_base); g_sn_base = NULL; g_sn_base_valid = 0;
}
static void q36_sn_init(Model *m, const char *dir, int bits) {
    const char *pack = getenv("COLI_STRATA_HEADS"), *trace = getenv("COLI_STRATA_TRACE");
    if ((!pack || !*pack) && (!trace || !*trace)) return;
    if (!dense_i8_on() || dense_bits()!=8 || !dense_idot_on()) {
        fprintf(stderr,"[strata] initial native qualification requires dense INT8 weights and integer-dot activations\n"); exit(1);
    }
    if (qt_ready() || g_cache_route || qq_active() || m->dn_dev ||
        g_q36_mux_slots > 1 || q36_has_clef_head(dir)) {
        fprintf(stderr,"[strata] heads/trace require qualified CPU text with one KV slot\n"); exit(1);
    }
#ifdef COLI_VULKAN
    if (g_vk_ready) { fprintf(stderr,"[strata] Vulkan head adapter pending\n"); exit(1); }
#endif
    g_sn_h = m->c.hidden;
    const char *input=getenv("COLI_STRATA_INPUT");
    if (input && *input && strcmp(input,"hidden") && strcmp(input,"next-embedding")) {
        fprintf(stderr,"[strata] COLI_STRATA_INPUT must be hidden or next-embedding\n"); exit(1);
    }
    g_sn_embedding=input && !strcmp(input,"next-embedding");
    g_sn_frontier = (float *)malloc((size_t)g_sn_h*sizeof(float));
    g_sn_rows = (float *)malloc((size_t)6*g_sn_h*sizeof(float));
    if (g_sn_embedding) g_sn_features=(float *)malloc((size_t)2*g_sn_h*sizeof(float));
    if (!g_sn_frontier || !g_sn_rows || (g_sn_embedding && !g_sn_features)) { q36_sn_shutdown(); fprintf(stderr,"[strata] hidden allocation failed\n"); exit(1); }
    unsigned char fingerprint[32]; char precision[160];
    snprintf(precision,sizeof precision,"qwen36:DBITS=8;IDOT=1;kernel=cpu;tap=%s/v1",
                g_sn_embedding?"final-rmsnorm+next-embedding":"final-rmsnorm");
    sn_status status = sn_fingerprint_model(dir,precision,fingerprint);
    if (status!=SN_OK) { q36_sn_shutdown(); fprintf(stderr,"[strata] model fingerprint failed\n"); exit(1); }
    if (pack && *pack) {
        if (status == SN_OK) status = sn_open(pack,fingerprint,(size_t)m->c.hidden*(g_sn_embedding?2:1),(size_t)m->c.vocab,&g_sn_bank);
        if (status != SN_OK || sn_get_info(g_sn_bank,&g_sn_info) != SN_OK) {
            q36_sn_shutdown(); fprintf(stderr,"[strata] pack refused (status=%d)\n",(int)status); exit(1);
        }
        g_sn_residual = sn_bank_kind(g_sn_bank) == SN_KIND_LOGIT_RESIDUAL;
        if (g_sn_residual) {
            g_sn_base = (float *)malloc((size_t)m->c.vocab*sizeof(float));
            if (!g_sn_base) { q36_sn_shutdown(); fprintf(stderr,"[strata] raw logit allocation failed\n"); exit(1); }
        }
        fprintf(stderr,"[strata] loaded %zu independent heads, source bits %u, decoded bytes %zu, kernel %s\n",
                g_sn_info.head_count,g_sn_info.source_bits,g_sn_info.decoded_bytes,g_sn_info.kernel);
    }
    if (trace && *trace) {
        g_sn_trace = fopen(trace,"wx");
        if (!g_sn_trace) { q36_sn_shutdown(); perror("COLI_STRATA_TRACE"); exit(1); }
        fprintf(g_sn_trace,"{\"schema\":\"colibri.native-hidden-raw/v1\",\"native_fingerprint\":\"");
        for(int j=0;j<32;j++) fprintf(g_sn_trace,"%02x",fingerprint[j]);
        fprintf(g_sn_trace,"\",\"quantization\":\"%s\",\"hidden_tap\":\"%s/v1\",\"hidden_size\":%d,\"native_hidden_size\":%d,\"vocab_size\":%d}\n",
                precision,g_sn_embedding?"final-rmsnorm+next-embedding":"final-rmsnorm",g_sn_h*(g_sn_embedding?2:1),g_sn_h,m->c.vocab);
    }
    atexit(q36_sn_shutdown);
}
static const float *q36_sn_input(Model *m,int chosen) {
    if (!g_sn_embedding) return g_sn_frontier;
    memcpy(g_sn_features,g_sn_frontier,(size_t)g_sn_h*sizeof(float));
    if (m->embed) memcpy(g_sn_features+g_sn_h,m->embed+(size_t)chosen*g_sn_h,(size_t)g_sn_h*sizeof(float));
    else for (int j=0;j<g_sn_h;j++) g_sn_features[g_sn_h+j]=f16_to_f32(m->embed_h[(size_t)chosen*g_sn_h+j]);
    return g_sn_features;
}
static void q36_sn_trace_known(Model *m,int chosen,int pos) {
    if (!g_sn_trace || !g_sn_trace_active || !g_sn_embedding || !g_sn_valid || g_sn_pos!=pos-1) return;
    const float *features=q36_sn_input(m,chosen);
    fprintf(g_sn_trace,"{\"session\":%llu,\"position\":%d,\"input_token\":%d,\"next_token\":%d,\"hidden\":[",
            g_sn_generation,g_sn_pos,g_sn_frontier_token,chosen);
    for(int j=0;j<2*g_sn_h;j++) fprintf(g_sn_trace,"%s%.9g",j?",":"",(double)features[j]);
    if(fprintf(g_sn_trace,"]}\n")<0 || fflush(g_sn_trace)) { fprintf(stderr,"[strata] trace write failed\n"); exit(1); }
}
static void q36_sn_capture(Model *m,const float *x,const int *ids,int S,int pos) {
    if (!g_sn_frontier || S < 1) return;
    if (g_sn_residual) g_sn_base_valid = 0;
    int D=m->c.hidden;
    if (g_sn_capture_rows == S && S <= 6) {
        for (int r=0;r<S;r++)
            rmsnorm_row(g_sn_rows+(size_t)r*D,x+(size_t)r*D,m->final_norm,D,m->c.eps);
    } else {
        rmsnorm_row(g_sn_frontier,x+(size_t)(S-1)*D,m->final_norm,D,m->c.eps);
        g_sn_valid=1; g_sn_pos=pos+S-1; g_sn_frontier_token=ids[S-1];
        if (g_sn_trace && g_sn_trace_active && S==1 && !g_sn_embedding) {
            fprintf(g_sn_trace,"{\"session\":%llu,\"position\":%d,\"input_token\":%d,\"hidden\":[",g_sn_generation,pos,ids[0]);
            for (int j=0;j<D;j++) fprintf(g_sn_trace,"%s%.9g",j?",":"",(double)g_sn_frontier[j]);
            if (fprintf(g_sn_trace,"]}\n")<0 || fflush(g_sn_trace)) { fprintf(stderr,"[strata] trace write failed\n"); exit(1); }
        }
    }
}
static void q36_sn_publish_logits(Model *m,const float *logit,int S,int nlogits,int pos) {
    if (!g_sn_base || !logit || S < 1 || nlogits < 1 || nlogits > S) return;
    /* Verify row zero is the committed frontier; later rows remain speculative.
     * A plain prefill/step captures its last row and its matching last logits. */
    int verify = g_sn_capture_rows == S;
    if (verify && nlogits != S) { g_sn_base_valid = 0; return; }
    int row = verify ? 0 : nlogits - 1;
    memcpy(g_sn_base,logit+(size_t)row*m->c.vocab,(size_t)m->c.vocab*sizeof(float));
    g_sn_base_pos = verify ? pos : pos+S-1;
    g_sn_base_valid = 1;
}
#endif
