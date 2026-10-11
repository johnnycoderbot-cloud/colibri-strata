/* Bounded native media preprocessing. No allocation, codecs, capture, model
 * execution, or device access. Every buffer remains caller-owned. */
#ifndef COLIBRI_NATIVE_MEDIA_H
#define COLIBRI_NATIVE_MEDIA_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum { NM_OK=0, NM_INVALID=1, NM_BOUNDS=2, NM_ALIAS=3, NM_UNSUPPORTED=4 } nm_status;
typedef enum { NM_WAV_PCM16=1, NM_WAV_FLOAT32=3 } nm_wav_encoding;
typedef struct {
    size_t sample_count;
    uint32_t sample_rate, channels;
    nm_wav_encoding encoding;
    double duration_seconds;
} nm_wav_info;

#define NM_WAV_DEFAULT_BYTES ((size_t)2*1024*1024)
#define NM_WAV_MAX_BYTES ((size_t)32*1024*1024)
#define NM_IMAGE_MAX_PIXELS ((size_t)16*1024*1024)
#define NM_SR_MAX_TILE_PIXELS ((size_t)270*480)

const char *nm_status_string(nm_status status);

/* Canonical RIFF/WAVE, one fmt then one data chunk, mono16k PCM16/IEEE F32.
 * max_seconds must be finite in (0,30]; actual samples are strictly <30sec.
 * max_bytes is explicit and <=32MiB. Metadata and samples are unchanged on
 * every error, including malformed/nonfinite samples or insufficient space.
 * Input may be unaligned; typed float/metadata outputs must be aligned. */
nm_status nm_wav_inspect(const void *blob, size_t bytes, size_t max_bytes,
                        double max_seconds, nm_wav_info *info);
nm_status nm_wav_decode(const void *blob, size_t bytes, size_t max_bytes,
                       double max_seconds, float *samples, size_t sample_capacity,
                       nm_wav_info *info);

/* Static model tile: dimensions1..1024 and <=270*480 pixels. Scratch size is
 * 15 bytes/pixel. start_x/y may be negative (halo). Pixels outside the decoded
 * source use EDGE REPLICATION, matching stream_models.py. Source is RGB/RGBA
 * HWC; alpha is ignored here. low/high are contiguous NCHW BGR F32 in0..255;
 * high is the Pillow-compatible bicubic4x second model INPUT, not an upscale
 * inference result. Capacities count float elements, workspace counts bytes.
 * All output/scratch/source ranges must be disjoint; failure writes nothing. */
nm_status nm_sr_workspace_bytes(size_t tile_width, size_t tile_height, size_t *bytes);
nm_status nm_sr_prepare(const uint8_t *source, size_t source_bytes,
                       size_t width, size_t height, size_t channels, size_t row_stride,
                       int64_t start_x, int64_t start_y, size_t tile_width, size_t tile_height,
                       float *low_bgr, size_t low_capacity, float *high_bgr, size_t high_capacity,
                       void *workspace, size_t workspace_bytes);

/* Finite normalized NCHW BGR model scores -> HWC RGB bytes. F32 clip*255 then
 * truncation matches numpy.astype(uint8), including0.25->63. Output row padding
 * is retained. Every source score is checked before any output is written. */
nm_status nm_bgr_output_rgb(const float *scores, size_t score_count,
                            size_t width, size_t height, uint8_t *rgb,
                            size_t rgb_bytes, size_t rgb_row_stride);

/* Pillow BICUBIC4x for one-channel alpha or three-channel RGB U8 arrays.
 * Pixel limit applies to the scaled output. Separable8bpc filtering uses
 * center mapping, cubic a=-0.5 and intermediate U8 rounding/clipping.
 * Scratch is the horizontal intermediate: width*4*height*channels bytes.
 * Caller-owned source/destination/scratch must be disjoint; no allocations. */
nm_status nm_bicubic4_workspace_bytes(size_t width, size_t height, size_t channels,
                                     size_t *bytes);
nm_status nm_bicubic4_u8(const uint8_t *source, size_t source_bytes,
                        size_t width, size_t height, size_t channels, size_t source_row_stride,
                        uint8_t *output, size_t output_bytes, size_t output_row_stride,
                        void *workspace, size_t workspace_bytes);

#ifdef __cplusplus
}
#endif
#endif
