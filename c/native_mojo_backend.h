#ifndef COLIBRI_NATIVE_MOJO_BACKEND_H
#define COLIBRI_NATIVE_MOJO_BACKEND_H
/* C model/tensor boundary to the native Modular MAX runtime and its Mojo
 * kernels. No Python inference. Compile against the matching MAX C SDK.
 * A backend instance is externally serialized. Inputs remain owned by caller;
 * inputs are shadow-copied before SDK borrowing; outputs are copied into
 * C-owned host storage and published transactionally.
 * Model state must be explicit graph inputs/outputs; no private SDK layout use. */
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define CMJ_MAX_TENSORS 32u
#define CMJ_MAX_RANK 8u
#define CMJ_MAX_BYTES ((size_t)128<<20)
typedef struct CMJEngine CMJEngine;
typedef enum { CMJ_F32=1,CMJ_I8,CMJ_I16,CMJ_I4,CMJ_I64,CMJ_I32,CMJ_U8,CMJ_F16,CMJ_BF16 } CMJDType;
typedef enum { CMJ_OK=0,CMJ_INVALID,CMJ_LIMIT,CMJ_DEVICE,CMJ_SDK,CMJ_NOMEM } CMJStatus;
typedef struct {
    const char *name;
    CMJDType dtype;
    size_t rank;
    int64_t shape[CMJ_MAX_RANK];
    const void *data;
    size_t bytes;
} CMJInput;
typedef struct {
    char name[129];
    CMJDType dtype;
    size_t rank;
    int64_t shape[CMJ_MAX_RANK];
    void *data;
    size_t bytes;
    int source_accelerator, source_device_id;
} CMJOutput;
typedef struct {
    /* -1 selects CPU. >=0 selects the exact SDK accelerator ID. No fallback. */
    int accelerator_id;
    const char *model_path; /* compatible immutable exported MAX .mef */
    size_t result_limit_bytes;
    const char *const *weight_paths; /* optional safetensor shards, exact graph names */
    size_t weight_count;
} CMJOptions;
const char *cmj_version(void); /* borrowed SDK string */
int cmj_accelerator_count(void);
int cmj_host_available(void);
/* Allocation-free storage admission query; bytes unchanged on failure. */
CMJStatus cmj_tensor_bytes(CMJDType,size_t,const int64_t *,size_t *);
CMJStatus cmj_create(const CMJOptions *,CMJEngine **,char *,size_t);
void cmj_destroy(CMJEngine *);
/* out must hold output_count entries; names must be distinct. Error leaves
 * out unchanged. On success each data allocation belongs to caller. I4/I8/I16/
 * I32/I64 are signed; U8 is unsigned. INT4 is a low-nibble-first storage codec expanded
 * to INT8 SDK input; its graph must declare INT8. Native INT4 arithmetic is
 * not claimed. Raw native INT4 outputs are refused until layout qualification.
 * Device proof covers source output-tensor placement, not every graph node. */
CMJStatus cmj_run(CMJEngine *,const CMJInput *,size_t,
                  const char *const *,size_t,CMJOutput *,char *,size_t);
void cmj_output_free(CMJOutput *,size_t);
#ifdef __cplusplus
}
#endif
#endif
