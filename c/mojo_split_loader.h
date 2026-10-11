#ifndef COLIBRI_MOJO_SPLIT_LOADER_H
#define COLIBRI_MOJO_SPLIT_LOADER_H
#include <stddef.h>
#include "runtime_split.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Optional native Mojo common planner, not a device kernel backend. Calls and
 * closure must be externally serialized. The returned planner is borrowed and
 * expires at close; retain this handle for every call. No Python inference.
 * NULL path selects COLI_MOJO_DISPATCH. An unset/invalid request is an error:
 * the caller chooses whether to use the ordinary C planner when unrequested.
 * out must point to NULL; failure leaves out unchanged. Native Windows refuses
 * the currently unavailable Mojo shared-library ABI before loading anything.
 */
typedef struct ColiMojoSplit ColiMojoSplit;
int coli_mojo_split_open(const char *path, ColiMojoSplit **out,
                         char *error, size_t error_cap);
ColiSplitPlanner coli_mojo_split_planner(const ColiMojoSplit *);
void coli_mojo_split_close(ColiMojoSplit *);
#ifdef __cplusplus
}
#endif
#endif
