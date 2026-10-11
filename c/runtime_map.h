#ifndef COLIBRI_RUNTIME_MAP_H
#define COLIBRI_RUNTIME_MAP_H
/* Native, backend-neutral placement policy. This prices one work unit using
 * caller-owned measured profiles; it neither opens devices nor moves state.
 * Adapters supply exact work/model/numeric identities and retain all ownership. */
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define COLI_MAP_MAX_NODES 32u
#define COLI_MAP_MAX_LINKS 128u
#define COLI_MAP_MAX_BUFFERS 64u
#define COLI_MAP_NONE UINT32_MAX
typedef enum {
    COLI_MAP_SSD=1, COLI_MAP_RAM, COLI_MAP_CPU, COLI_MAP_GPU,
    COLI_MAP_NPU, COLI_MAP_TPU, COLI_MAP_AIPU
} ColiMapKind;
enum {
    COLI_MAP_INT4=1u<<0, COLI_MAP_INT8=1u<<1, COLI_MAP_INT16=1u<<2,
    COLI_MAP_F16=1u<<3, COLI_MAP_BF16=1u<<4, COLI_MAP_F32=1u<<5
};
enum {
    COLI_MAP_MOVABLE=1u<<0, COLI_MAP_FENCE_DONE=1u<<1,
    COLI_MAP_STATE=1u<<2, COLI_MAP_PREFETCH=1u<<3
};
typedef struct {
    uint32_t kind, execution_verified, dtypes;
    uint64_t budget_bytes; /* work-unit budget, including its resident inputs */
    uint64_t memory_domain; /* nonzero caller-issued shared physical pool identity */
    /* Caller-issued exact identity handles, not security hashes. Work includes
     * phase, shape and batch; numeric includes model/checkpoint/state encoding. */
    uint64_t work_id, numeric_id, valid_until_ns;
    double compute_us, queue_us, measured_overlap_us;
} ColiMapNode;
typedef struct {
    uint32_t from, to;
    uint64_t valid_until_ns;
    double bytes_per_second, latency_us;
    uint32_t zero_copy; /* verified alias in the same nonzero memory_domain */
} ColiMapLink;
typedef struct {
    uint32_t owner, flags;
    uint64_t bytes;
} ColiMapBuffer;
typedef struct {
    uint64_t now_ns, work_id, numeric_id, workspace_bytes, output_bytes;
    uint32_t dtypes, incumbent, output_owner;
    /* Migration must save this much over the incumbent's complete cost. */
    double minimum_saving_us;
    double maximum_latency_us; /* zero is uncapped; includes transfers/queue */
    const ColiMapNode *nodes;
    size_t node_count;
    const ColiMapLink *links;
    size_t link_count;
    const ColiMapBuffer *buffers;
    size_t buffer_count;
} ColiMapRequest;
typedef struct {
    uint32_t target, moved_state;
    uint64_t required_bytes, routed_bytes; /* payload bytes; staging not counted twice */
    double compute_us, queue_us, input_transfer_us, output_transfer_us;
    double overlap_credit_us, total_us;
} ColiMapDecision;
typedef enum { COLI_MAP_OK=0, COLI_MAP_INVALID, COLI_MAP_NO_ROUTE } ColiMapStatus;
/* All buffers must have stable ownership for this call. Directed shortest-time
 * transfer paths include SSD/RAM staging, latency and measured bandwidth.
 * Expired/unverified profiles are ineligible. State migration requires MOVABLE
 * and FENCE_DONE, with exact numeric identity. Overlap credit only applies to
 * explicitly PREFETCHable non-state inputs, bounded by measured overlap/compute.
 * Required bytes includes inputs+workspace+output; shared pools/concurrent
 * reservations must already be reflected in budget_bytes by the caller.
 * Estimates are not allocations, execution, or speedup evidence.
 * On error decision is unchanged; on success it is written once. */
ColiMapStatus coli_runtime_map(const ColiMapRequest *request, ColiMapDecision *decision);
#ifdef __cplusplus
}
#endif
#endif
