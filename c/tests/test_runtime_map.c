/* Policy fixtures use declared synthetic measurements. They demonstrate routing
 * correctness, not physical execution or speed. No model/device/SDK is loaded. */
#include "../runtime_map.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static ColiMapNode nodes[7];
static ColiMapLink links[8];
static ColiMapBuffer buffers[4];
static ColiMapRequest request;
static ColiMapDecision decision;
static unsigned checks;
static void near(double actual,double expected) {
    checks++;assert(fabs(actual-expected)<0.00001);
}
static void setup(void) {
    memset(nodes,0,sizeof nodes);memset(links,0,sizeof links);memset(buffers,0,sizeof buffers);
    memset(&request,0,sizeof request);memset(&decision,0,sizeof decision);
    for(unsigned k=0;k<7;k++) {
        nodes[k].kind=k+1;nodes[k].budget_bytes=1u<<20;
        nodes[k].execution_verified=k>=2;nodes[k].dtypes=COLI_MAP_INT4|COLI_MAP_INT8|COLI_MAP_INT16;
        nodes[k].work_id=11;nodes[k].numeric_id=22;nodes[k].valid_until_ns=100;
        nodes[k].compute_us=100;
    }
    request.now_ns=100;request.work_id=11;request.numeric_id=22;
    request.nodes=nodes;request.node_count=7;request.links=links;request.buffers=buffers;
    request.dtypes=COLI_MAP_INT4;request.incumbent=2;request.output_owner=1;
    request.minimum_saving_us=0;
}
static void edge(unsigned index,unsigned from,unsigned to,double bandwidth,double latency) {
    links[index]=(ColiMapLink){from,to,100,bandwidth,latency,0};
    if(request.link_count<=index)request.link_count=index+1;
}
static void expect(unsigned target) {
    checks++;assert(coli_runtime_map(&request,&decision)==COLI_MAP_OK);assert(decision.target==target);
}
static void refusal(ColiMapStatus status) {
    ColiMapDecision before;memset(&decision,0x5a,sizeof decision);before=decision;
    checks++;assert(coli_runtime_map(&request,&decision)==status);assert(!memcmp(&decision,&before,sizeof before));
}
static void all_compute_classes(void) {
    for(unsigned target=2;target<7;target++) {
        setup();nodes[target].compute_us=1;expect(target);near(decision.total_us,1);
    }
    setup();expect(2); /* Equal complete costs retain incumbent, no device bias. */
    nodes[0].execution_verified=1;refusal(COLI_MAP_INVALID); /* SSD is storage. */
}
static void migration_and_bandwidth(void) {
    setup();request.node_count=4;nodes[3].compute_us=10;
    buffers[0]=(ColiMapBuffer){2,COLI_MAP_STATE|COLI_MAP_MOVABLE|COLI_MAP_FENCE_DONE,10000};
    request.buffer_count=1;edge(0,2,3,10000000,0);
    expect(2); /* GPU compute is faster; one state transfer costs 1000us. */
    edge(0,2,3,1000000000,5);expect(3);near(decision.total_us,25);
    assert(decision.moved_state&&decision.routed_bytes==10000);
    buffers[0].flags&=~COLI_MAP_FENCE_DONE;expect(2);
    buffers[0].flags=COLI_MAP_STATE|COLI_MAP_FENCE_DONE;expect(2); /* Pinned. */
    buffers[0].flags=COLI_MAP_STATE|COLI_MAP_MOVABLE|COLI_MAP_FENCE_DONE;
    nodes[3].numeric_id=23;expect(2);nodes[3].numeric_id=22;
    nodes[3].work_id=12;expect(2);nodes[3].work_id=11;
    nodes[3].dtypes=COLI_MAP_INT8;expect(2);nodes[3].dtypes=COLI_MAP_INT4;
    nodes[3].execution_verified=0;expect(2);nodes[3].execution_verified=1;
    nodes[3].valid_until_ns=99;expect(2);nodes[3].valid_until_ns=100;
    links[0].valid_until_ns=99;expect(2);links[0].valid_until_ns=100;
    request.minimum_saving_us=75;expect(2); /* 100-25, threshold is strict. */
    request.minimum_saving_us=74;expect(3);
    nodes[3].queue_us=100;expect(2);
    nodes[3].queue_us=0;request.minimum_saving_us=1000;
    request.maximum_latency_us=25;expect(3); /* Incumbent misses deadline. */
    request.maximum_latency_us=24;refusal(COLI_MAP_NO_ROUTE);
}
static void staged_ssd_and_outputs(void) {
    setup();request.node_count=4;nodes[2].execution_verified=0;nodes[3].compute_us=20;
    buffers[0]=(ColiMapBuffer){0,COLI_MAP_MOVABLE|COLI_MAP_FENCE_DONE,1000};
    request.buffer_count=1;
    edge(0,0,1,10000000,10);edge(1,1,3,100000000,2);
    expect(3);near(decision.input_transfer_us,122);near(decision.total_us,142);
    request.output_bytes=1000;
    refusal(COLI_MAP_NO_ROUTE); /* No invented reverse edge for output. */
    edge(2,3,1,100000000,3);expect(3);near(decision.output_transfer_us,13);
    near(decision.total_us,155);assert(decision.routed_bytes==2000);
    nodes[1].budget_bytes=999;refusal(COLI_MAP_NO_ROUTE);
    nodes[1].budget_bytes=1000;
    edge(3,0,3,10000000,0);expect(3);near(decision.input_transfer_us,100);
    nodes[3].budget_bytes=1999;refusal(COLI_MAP_NO_ROUTE);
    nodes[3].budget_bytes=2000;expect(3);assert(decision.required_bytes==2000);
    request.workspace_bytes=1;refusal(COLI_MAP_NO_ROUTE);
}
static void credit_and_types(void) {
    setup();request.node_count=4;nodes[2].execution_verified=0;nodes[3].compute_us=20;
    buffers[0]=(ColiMapBuffer){1,COLI_MAP_PREFETCH|COLI_MAP_MOVABLE|COLI_MAP_FENCE_DONE,1000};
    request.buffer_count=1;edge(0,1,3,10000000,0);nodes[3].measured_overlap_us=10;
    expect(3);near(decision.total_us,110);near(decision.overlap_credit_us,10);
    nodes[3].measured_overlap_us=1000;expect(3);near(decision.overlap_credit_us,20);
    buffers[0].flags&=~COLI_MAP_PREFETCH;expect(3);near(decision.overlap_credit_us,0);
    buffers[0].flags|=COLI_MAP_STATE;expect(3);near(decision.overlap_credit_us,0);
    buffers[0].flags|=COLI_MAP_PREFETCH;refusal(COLI_MAP_INVALID);
    setup();
    for(unsigned dtype=COLI_MAP_INT4;dtype<=COLI_MAP_INT16;dtype<<=1) {
        request.dtypes=dtype;nodes[6].compute_us=1;expect(6);
    }
    request.dtypes=COLI_MAP_INT4|COLI_MAP_INT16;nodes[6].dtypes=COLI_MAP_INT4;expect(2);
}
static void malformed_and_overflow(void) {
    setup();nodes[2].compute_us=NAN;refusal(COLI_MAP_INVALID);
    setup();nodes[3].compute_us=INFINITY;refusal(COLI_MAP_INVALID);
    setup();nodes[4].queue_us=-1;refusal(COLI_MAP_INVALID);
    setup();request.minimum_saving_us=NAN;refusal(COLI_MAP_INVALID);
    setup();request.maximum_latency_us=INFINITY;refusal(COLI_MAP_INVALID);
    setup();request.numeric_id=0;refusal(COLI_MAP_INVALID);
    setup();request.node_count=COLI_MAP_MAX_NODES+1;refusal(COLI_MAP_INVALID);
    setup();request.buffer_count=COLI_MAP_MAX_BUFFERS+1;refusal(COLI_MAP_INVALID);
    setup();request.dtypes=1u<<31;refusal(COLI_MAP_INVALID);
    setup();request.output_bytes=1;request.output_owner=99;refusal(COLI_MAP_INVALID);
    setup();edge(0,1,3,0,0);refusal(COLI_MAP_INVALID);
    setup();edge(0,1,3,NAN,0);refusal(COLI_MAP_INVALID);
    setup();edge(0,1,3,1,INFINITY);refusal(COLI_MAP_INVALID);
    setup();buffers[0]=(ColiMapBuffer){99,0,1};request.buffer_count=1;refusal(COLI_MAP_INVALID);
    setup();request.workspace_bytes=UINT64_MAX;request.output_bytes=1;refusal(COLI_MAP_NO_ROUTE);
    setup();nodes[2].compute_us=DBL_MAX;nodes[2].queue_us=DBL_MAX;
    for(unsigned k=3;k<7;k++)nodes[k].execution_verified=0;
    refusal(COLI_MAP_NO_ROUTE);
    setup();request.incumbent=COLI_MAP_NONE;nodes[2].execution_verified=0;expect(3);
}
static void shared_memory(void) {
    setup();request.node_count=4;nodes[3].compute_us=10;
    buffers[0]=(ColiMapBuffer){2,COLI_MAP_MOVABLE|COLI_MAP_FENCE_DONE|COLI_MAP_STATE,10000};
    request.buffer_count=1;edge(0,2,3,0,3);links[0].zero_copy=1;
    refusal(COLI_MAP_INVALID); /* Shared access must have explicit pool identity. */
    nodes[2].memory_domain=5;nodes[3].memory_domain=6;refusal(COLI_MAP_INVALID);
    nodes[3].memory_domain=5;expect(3);near(decision.total_us,13);
    assert(decision.moved_state&&decision.routed_bytes==10000);
    buffers[0].flags&=~COLI_MAP_FENCE_DONE;expect(2); /* Aliasing does not bypass fences. */
    buffers[0].flags|=COLI_MAP_FENCE_DONE;nodes[3].numeric_id=23;expect(2);
}
int main(void) {
    all_compute_classes();migration_and_bandwidth();staged_ssd_and_outputs();
    credit_and_types();malformed_and_overflow();shared_memory();
    printf("native state/bandwidth mapping: %u policy checks passed (synthetic profiles, no hardware speed claim)\n",checks);
    return 0;
}
