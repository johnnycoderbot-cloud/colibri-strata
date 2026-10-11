/* C-only OpenVINO/GenAI client boundary. No Python inference calls.
 * Each engine is externally serialized; destroy only after its call completes.
 * SDK paths name trusted native shared libraries. Missing C SDK symbols are
 * reported as a missing component, never substituted by another ML backend.
 */
#ifndef COLI_NATIVE_GRAPH_BACKEND_H
#define COLI_NATIVE_GRAPH_BACKEND_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct CNGEngine CNGEngine;
typedef enum { CNG_TEXT=1, CNG_AUDIO=2, CNG_CAMERA=3 } CNGModality;
typedef enum {
    CNG_OK=0, CNG_INVALID, CNG_LIMIT, CNG_NO_SDK, CNG_DEVICE,
    CNG_SDK_ERROR, CNG_NONFINITE, CNG_NOMEM, CNG_UNSUPPORTED
} CNGStatus;
typedef struct {
    const char *model_dir;
    CNGModality modality;
    const char *device;              /* explicit plugin/device; AUTO/MULTI refused */
    const char *ov_c_library;
    const char *genai_c_library;     /* required only for TEXT/AUDIO */
    const char *tokenizers_library; /* TEXT extension; NULL derives from GenAI SDK dir */
    const char *sr_xml;              /* required only for CAMERA, actual trained IR */
    unsigned max_tokens;           /* TEXT1..256(default256); AUDIO1..448(default448) */
    unsigned max_prompt_tokens;    /* TEXT1..4096; zero uses2048 */
    unsigned max_context_tokens;   /* TEXT2..4096; zero uses4096 */
    int inference_f16;             /* SR CPU/GPU precision hint, not INT16 */
} CNGOptions;
typedef struct {
    CNGModality modality;
    char requested_device[64];
    char selected_device[128];
    int execution_devices_verified; /* graph compiled EXECUTION_DEVICES proof only */
    char runtime_precision[64];     /* SDK-reported hint, or unknown */
    size_t lr_width, lr_height, bicubic_width, bicubic_height;
    size_t output_width, output_height;
    unsigned scale, max_tokens, max_prompt_tokens, max_context_tokens;
} CNGInfo;

/* Output pointers must initially be NULL. Outputs/counts/info are published
 * only on success; errors are NUL-terminated when error_capacity>0. Outputs
 * use malloc and may be freed with cng_free() or the same C runtime's free(). */
CNGStatus cng_available_devices(const char *ov_c_library,char **utf8_lines,
                                size_t *bytes,char *error,size_t error_capacity);
CNGStatus cng_create(const CNGOptions *options,CNGEngine **engine,
                     char *error,size_t error_capacity);
void cng_destroy(CNGEngine *engine);
CNGStatus cng_info(const CNGEngine *engine,CNGInfo *info);
CNGStatus cng_last_prompt_tokens(const CNGEngine *engine,size_t *tokens);
/* Native model tokenizer chat template, add_generation_prompt=true. Roles
 * are system/user/assistant; 1..64 messages, valid NUL-terminated UTF8 strings,
 * total content <=1MiB. Render once, then call raw cng_text for exact bounds. */
CNGStatus cng_render_chat(CNGEngine *engine,const char *const *roles,
                          const char *const *contents,size_t message_count,
                          char **utf8,size_t *bytes,char *error,size_t error_capacity);

/* UTF8 prompt is stateless, nonempty, NUL-free and <=1MiB. max_tokens is
 * 1..engine limit (zero uses that limit). Native SDK tokenizer enforces both
 * prompt and prompt+generation context limits. bytes excludes trailing NUL. */
CNGStatus cng_text(CNGEngine *engine,const char *prompt,size_t prompt_bytes,
                   unsigned max_tokens,char **utf8,size_t *bytes,
                   char *error,size_t error_capacity);
/* Independent mono16k audio, 1..479999 finite normalized samples (<30s).
 * task is transcribe or translate; language is NULL(auto) or en.
 * translate targets English. The SDK retains no shared chat/session state. */
CNGStatus cng_whisper(CNGEngine *engine,const float *samples,size_t sample_count,
                      const char *task,const char *language,char **utf8,size_t *bytes,
                      char *error,size_t error_capacity);
/* Exact learned-model tile geometry comes from cng_info. Immutable planar
 * BGR inputs have3*width*height F32 values in0..255; bicubic is preprocessing,
 * not an output fallback. Returned planar BGR model output is finite F32,
 * conventionally0..1 for SISR1032, with exact output_count. */
CNGStatus cng_sr(CNGEngine *engine,const float *lr,size_t lr_count,
                 const float *bicubic,size_t bicubic_count,float **output,
                 size_t *output_count,char *error,size_t error_capacity);
void cng_free(void *owned_output);
const char *cng_status_string(CNGStatus status);
#ifdef __cplusplus
}
#endif
#endif
