#ifndef COLIBRI_NATIVE_GRAPH_MANIFEST_H
#define COLIBRI_NATIVE_GRAPH_MANIFEST_H
#include <stddef.h>
#include <stdint.h>
#include "native_graph_backend.h"

typedef struct {
    char root[4096], model_dir[4096], sr_xml[4096];
    char family[64], model_id[256], precision[65];
    CNGModality modality;
    uint64_t asset_bytes, base_estimated_bytes, memory_budget_bytes;
    size_t max_input_bytes, max_output_bytes, max_output_pixels;
    unsigned max_tokens, max_prompt_tokens, max_context_tokens;
    int tile_overlap, inference_f16;
    double max_audio_seconds;
} CGManifest;

/* Read the existing immutable model manifest and verify every declared file.
 * Admission values are estimates, not a reservation of physical device RAM.
 * Asset names are portable ASCII relative paths; SDK/model roots are UTF8.
 * Outputs remain untouched on failure. No SDK, Python or device is loaded. */
int cg_manifest_load(const char *root, uint64_t budget, CGManifest *out,
                     char *error, size_t error_capacity);
#endif
