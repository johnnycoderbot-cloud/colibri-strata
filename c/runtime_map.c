#include "runtime_map.h"
#include <float.h>
#include <math.h>
#include <string.h>
_Static_assert(sizeof(double)==sizeof(uint64_t)&&DBL_MANT_DIG==53&&DBL_MAX_EXP==1024,"binary64 cost model required");

#define DTYPES (COLI_MAP_INT4|COLI_MAP_INT8|COLI_MAP_INT16|COLI_MAP_F16|COLI_MAP_BF16|COLI_MAP_F32)
#define FLAGS (COLI_MAP_MOVABLE|COLI_MAP_FENCE_DONE|COLI_MAP_STATE|COLI_MAP_PREFETCH)
static int finite_nonnegative(double x) {
    /* Bit inspection retains rejection under the parent's fast-math builds. */
    uint64_t bits=0;memcpy(&bits,&x,sizeof bits);
    return (bits&UINT64_C(0x7ff0000000000000))!=UINT64_C(0x7ff0000000000000)&&x>=0;
}
static int add_u64(uint64_t *a,uint64_t b) {
    if(b>UINT64_MAX-*a)return 0;*a+=b;return 1;
}
static int valid(const ColiMapRequest *r) {
    if(!r||!r->nodes||!r->node_count||r->node_count>COLI_MAP_MAX_NODES||
       r->link_count>COLI_MAP_MAX_LINKS||r->buffer_count>COLI_MAP_MAX_BUFFERS||
       (r->link_count&&!r->links)||(r->buffer_count&&!r->buffers)||
       !r->work_id||!r->numeric_id||!r->dtypes||(r->dtypes&~DTYPES)||
       (r->incumbent!=COLI_MAP_NONE&&r->incumbent>=r->node_count)||
       (r->output_bytes&&r->output_owner>=r->node_count)||
       !finite_nonnegative(r->minimum_saving_us)||!finite_nonnegative(r->maximum_latency_us))return 0;
    for(size_t n=0;n<r->node_count;n++) {
        const ColiMapNode *d=r->nodes+n;
        if(d->kind<COLI_MAP_SSD||d->kind>COLI_MAP_AIPU||d->execution_verified>1||
           (d->dtypes&~DTYPES)||!finite_nonnegative(d->compute_us)||
           !finite_nonnegative(d->queue_us)||!finite_nonnegative(d->measured_overlap_us)||
           ((d->kind==COLI_MAP_SSD||d->kind==COLI_MAP_RAM)&&d->execution_verified))return 0;
    }
    for(size_t n=0;n<r->link_count;n++) {
        const ColiMapLink *l=r->links+n;
        if(l->from>=r->node_count||l->to>=r->node_count||l->from==l->to||
           !finite_nonnegative(l->bytes_per_second)||(!l->zero_copy&&!l->bytes_per_second)||
           !finite_nonnegative(l->latency_us)||l->zero_copy>1||
           (l->zero_copy&&(!r->nodes[l->from].memory_domain||
              r->nodes[l->from].memory_domain!=r->nodes[l->to].memory_domain)))return 0;
    }
    for(size_t n=0;n<r->buffer_count;n++) {
        const ColiMapBuffer *b=r->buffers+n;
        if(b->owner>=r->node_count||(b->flags&~FLAGS)||
           ((b->flags&COLI_MAP_STATE)&&(b->flags&COLI_MAP_PREFETCH)))return 0;
    }return 1;
}
static double transfer(const ColiMapRequest *r,uint32_t from,uint32_t to,uint64_t bytes) {
    if(from==to||!bytes)return 0;
    double costs[COLI_MAP_MAX_NODES];unsigned char done[COLI_MAP_MAX_NODES]={0};
    for(size_t n=0;n<r->node_count;n++)costs[n]=DBL_MAX;costs[from]=0;
    for(size_t n=0;n<r->node_count;n++) {
        uint32_t at=COLI_MAP_NONE;
        for(uint32_t k=0;k<r->node_count;k++)if(!done[k]&&(at==COLI_MAP_NONE||costs[k]<costs[at]))at=k;
        if(at==COLI_MAP_NONE||costs[at]==DBL_MAX)return DBL_MAX;
        if(at==to)return costs[at];done[at]=1;
        for(size_t k=0;k<r->link_count;k++) {
            const ColiMapLink *l=r->links+k;
            if(l->from!=at||done[l->to]||l->valid_until_ns<r->now_ns)continue;
            if(bytes>r->nodes[l->to].budget_bytes)continue;
            /* Serial staging: no invented SSD->device direct-transfer path. */
            double cost=l->latency_us;
            if(!l->zero_copy)cost+=(double)bytes/l->bytes_per_second*1000000.0;
            double next=costs[at]+cost;
            if(finite_nonnegative(next)&&next<costs[l->to])costs[l->to]=next;
        }
    }return DBL_MAX;
}
static int candidate(const ColiMapRequest *r,uint32_t at,ColiMapDecision *out) {
    const ColiMapNode *d=r->nodes+at;
    if(!d->execution_verified||d->kind<COLI_MAP_CPU||d->valid_until_ns<r->now_ns||
       d->work_id!=r->work_id||d->numeric_id!=r->numeric_id||
       (d->dtypes&r->dtypes)!=r->dtypes)return 0;
    ColiMapDecision c={0};c.target=at;c.required_bytes=r->workspace_bytes;
    if(!add_u64(&c.required_bytes,r->output_bytes))return 0;
    double prefetch=0;
    for(size_t k=0;k<r->buffer_count;k++) {
        const ColiMapBuffer *b=r->buffers+k;
        if(!add_u64(&c.required_bytes,b->bytes))return 0;
        if(!b->bytes||b->owner==at)continue;
        if((b->flags&(COLI_MAP_MOVABLE|COLI_MAP_FENCE_DONE))!=(COLI_MAP_MOVABLE|COLI_MAP_FENCE_DONE))return 0;
        double cost=transfer(r,b->owner,at,b->bytes);
        if(cost==DBL_MAX||!add_u64(&c.routed_bytes,b->bytes))return 0;
        c.input_transfer_us+=cost;
        if(b->flags&COLI_MAP_STATE)c.moved_state=1;
        if(b->flags&COLI_MAP_PREFETCH)prefetch+=cost;
    }
    if(c.required_bytes>d->budget_bytes)return 0;
    c.output_transfer_us=transfer(r,at,r->output_owner,r->output_bytes);
    if(c.output_transfer_us==DBL_MAX)return 0;
    if(r->output_bytes&&r->output_owner!=at&&!add_u64(&c.routed_bytes,r->output_bytes))return 0;
    c.compute_us=d->compute_us;c.queue_us=d->queue_us;
    c.overlap_credit_us=fmin(prefetch,fmin(d->compute_us,d->measured_overlap_us));
    c.total_us=c.input_transfer_us+c.output_transfer_us+c.compute_us+c.queue_us-c.overlap_credit_us;
    if(!finite_nonnegative(c.input_transfer_us)||!finite_nonnegative(prefetch)||!finite_nonnegative(c.total_us))return 0;
    if(r->maximum_latency_us&&c.total_us>r->maximum_latency_us)return 0;
    *out=c;return 1;
}
ColiMapStatus coli_runtime_map(const ColiMapRequest *r,ColiMapDecision *out) {
    if(!out||!valid(r))return COLI_MAP_INVALID;
    ColiMapDecision best={0},current={0};int found=0,have_current=0;
    for(uint32_t at=0;at<r->node_count;at++) {
        ColiMapDecision c;
        if(!candidate(r,at,&c))continue;
        if(at==r->incumbent){current=c;have_current=1;}
        if(!found||c.total_us<best.total_us||
           (c.total_us==best.total_us&&at==r->incumbent)){best=c;found=1;}
    }
    if(!found)return COLI_MAP_NO_ROUTE;
    if(have_current&&best.target!=current.target&&
       current.total_us-best.total_us<=r->minimum_saving_us)best=current;
    *out=best;return COLI_MAP_OK;
}
