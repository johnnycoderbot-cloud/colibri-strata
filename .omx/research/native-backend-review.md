# Native C backend independent review

Date: 2026-10-11. Reviewer: `/root/architecture`. Verdict: **approved** for the reviewed C backend source slice. Native workload/device qualification remains a separate parent gate; this approval does not claim trained-model results or acceleration.

## Findings

No unresolved blocking source finding in the reviewed version. Three earlier actionable boundary findings are resolved:

1. Whisper formerly accepted finite samples outside the declared normalized range. `c/native_graph_backend.c:433-434` now rejects samples outside `[-1,1]`; positive/negative out-of-range regressions are in `c/tests/test_native_graph_backend.c:150`.
2. The Windows temporary-configuration reopen path could follow a replaced/reparse file, and did not correctly bound `GetTempPathA`'s required length. That path has been removed entirely. Text and Whisper now take owned pipeline configuration copies and apply declared SDK setters (`c/native_graph_backend.c:417-422`, `:436-439`).
3. The backend's independent source-configuration reader lacked the full regular-file/no-reparse protections required for immutable assets. That reader and all JSON serialization/file handling have also been removed. The process's manifest loader owns asset validation; the backend delegates model loading to the official SDK without rewriting model assets.

## Source and ABI review

Reviewed `c/native_graph_backend.h`, `c/native_graph_backend.c`, and the fake-SDK tests against the matching native SDK **2026.4.1.0**, official source `ab01d1771e4adaa0e1e08e9d630def170572506c`.

| Boundary | Reviewed behavior |
|---|---|
| C ABI | Opaque handles only; public scalar/function ABI; C shape `{int64_t rank; int64_t *dims;}` and device-list layout match official headers. F32=4, I64=10, STRING=23 are correct. Variadic property counts use two arguments per key/value. No custom C++ object-layout access. |
| Text configuration | Owned pipeline copy preserves model EOS/options. Sets explicit token ceiling, greedy sampling, one beam/return sequence and `apply_chat_template=false`. One result sequence avoids the SDK's multi-result score formatter. |
| Native template | `cng_render_chat` binds the approved const C renderer, validates 1..64 system/user/assistant UTF-8 messages and bounded aggregate content, then reads a bounded caller-owned buffer. Production orchestration must render once before raw generation. |
| Token admission | Executes the model's tokenizer IR through the core C SDK, validates I64 `[1,N]`, then refuses prompt or prompt+generation excess before LLM generation. Default prompt/context bounds are 2048/4096; text generation is at most256. |
| Speech | Independent normalized mono16k samples, count1..479999, task transcribe/translate, language auto/en; owned model configuration copy plus direct Whisper max-new-token setter, at most448. Uses exact indexed text rather than human-formatted result conversion. |
| Learned SR | Requires static F32 planar BGR graph inputs0/1 and one output, bounded exact tile geometry, matching scale2..8 and finite data. Executes model inference and validates output shape/type/count before publication. |
| Device routing | Refuses AUTO/MULTI/HETERO/BATCH and absent explicit devices. Core SR checks exclusive `EXECUTION_DEVICES` before/after inference. GenAI constructors receive an explicit device but expose no compiled-model placement property; `execution_devices_verified=0` remains mandatory for those routes. |
| Cleanup/status | Frees owned config/results/tensors/shapes on error and success. Engine handles are destroyed before their DLLs unload. Failed calls do not publish output/count. Bounds and UTF-8 result integrity are checked before publication. Engine calls are externally serialized and destruction is permitted only after calls complete. |

No production Python inference call, model SDK Python binding, temporary model-configuration file or source-asset rewrite exists in this backend slice. DLL paths are trusted dependency inputs; immutable model assets are validated by the separate process manifest boundary.

## Verification evidence

Independent model-free run:

```text
gcc -std=c17 -O1 -g -Wall -Wextra -Werror -Wno-misleading-indentation \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    c/tests/test_native_graph_backend.c -lm -ldl \
    -o outputs-native/native-sdk-discovery/backend-review-asan
ASAN_OPTIONS=detect_leaks=1 ./outputs-native/native-sdk-discovery/backend-review-asan

native C backend fake-SDK: text4/audio2/SR6/tokenizer8 calls, limits,
device refusal, exact ownership passed
```

Ran in WSL `kali-pc` with exit0 and no ASan/UBSan/leak diagnostic. Tests cover malformed UTF-8, byte/token/context bounds, missing template setter, absent device/AUTO refusal, requested GPU.1 routing, configuration failure, malformed text size/NUL, SDK failure, SR metadata/type/shape/nonfinite output, post-inference CPU substitution and cleanup. Fake SDK values establish dispatch and ownership behavior only.

The separate matching SDK dependency build preserved all153 original exports and added exactly the three approved APIs. An MSVC pure-C C17 `/TC /W4 /WX` probe passed apply-template setter calls, exact Whisper max-token round-trip137, and renderer NULL rejection with buffer/size invariance. See `.omx/research/native-sdk-map.md`; this reviewer authored that dependency patch, so its independent review belongs to the parent.

Reviewed file SHA256 identities:

- `c/native_graph_backend.c`: `58565399c4b2524aedbcd65f066bd99d0cda1cd9566ee90fd32f2f22795d8816`
- `c/native_graph_backend.h`: `e1e666f9a42034d44656f8ed38e3c3e37b31e24c5af45e8b2fa8c06f5074bd38`
- `c/tests/test_native_graph_backend.c`: `188161c64131567cce7fe1dd1d896698620846ecf9ffea9d4a3aa6865a8a31c8`

## Remaining qualification limits

- Actual model template rendering, tokenizer special-token/count parity, trained neural output and native process/CLI/API framing are parent integration gates. This review does not substitute fake SDK values or ABI probes for those results.
- GenAI `selected_device` presently repeats the requested constructor device while its proof flag is false. Consumers must distinguish that request from SR's verified placement; displaying an unknown selected device until proof exists would be clearer.
- The public Whisper C generate API exposes no callback/cancel handle. Process-level timeout/cancellation must wait for completion or terminate its owned worker and suppress publication; it must not free a pipeline while inference runs.
- Hardware execution/precision/acceleration claims require real measured evidence. Installed plugins do not qualify a physical NPU/TPU/AIPU, and an inference precision hint does not prove arithmetic precision.
- Overall `.omx/state/native-graph-port-state.json` correctly remains active/in-progress. Source approval here completes only this reviewer handoff; workload integration/publication remains incomplete.
