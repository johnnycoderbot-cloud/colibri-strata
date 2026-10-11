#ifndef COLIBRI_RUNTIME_SPLIT_H
#define COLIBRI_RUNTIME_SPLIT_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define COLI_SPLIT_ABI 1u
#define COLI_SPLIT_MAX_EXPERTS 1024u
/* Flat ABI shared with native Mojo. controls[4]: ABI, misses, max_device,
 * transfer_overlap (0=serial staging before both branches, 1=transfer on device
 * branch). costs[9], in microseconds: host_per_expert, transfer_per_expert,
 * device_per_expert, host_base, device_base, host_queue, device_queue, join,
 * serial_base. These must be achieved costs for the current exact work,
 * numeric, device and contention regime; storage ceilings are not profiles.
 * counts[2]: device experts, host experts. times[4]: host, device, serial,
 * total including join. Each expert is executed once; hits remain in base work.
 * Candidate zero does not launch/charge a cold device branch unless base work
 * already requires it. Candidate ties prefer fewer transfers. No state moves.
 * All output entries are unchanged on failure. All costs must be finite >=0.
 * A zero host/device per-expert cost denotes an unavailable execution branch,
 * not free execution. max_device=0 makes the cold branch CPU-only.
 */
typedef int (*ColiSplitPlanner)(const uint32_t *,const double *,uint32_t *,double *);
int coli_split_plan(const uint32_t controls[4],const double costs[9],
                    uint32_t counts[2],double times[4]);
#ifdef __cplusplus
}
#endif
#endif
