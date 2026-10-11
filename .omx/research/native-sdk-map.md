# Native C SDK map — Windows, 2026-10-11

Status: native dependency discovery verified. No production workload was removed or disabled. Text, speech/English translation and trained image/video upscaling remain workloads to port and qualify through the C core.

## Matching official SDK

- OpenVINO GenAI **2026.4.1.0**, pinned source commit `ab01d1771e4adaa0e1e08e9d630def170572506c` ([official source](https://github.com/openvinotoolkit/openvino.genai/tree/ab01d1771e4adaa0e1e08e9d630def170572506c)).
- Official Windows x86-64 archive: [OpenVINO GenAI native SDK](https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.4.1/windows/openvino_genai_windows_2026.4.1.0_x86_64.zip).
- Downloaded archive: `outputs-native/native-sdk-discovery/openvino_genai_windows_2026.4.1.0_x86_64.zip`, 254,143,615 bytes. Measured SHA256 `437dc85e0fa60d2231b28f323c9b06ee27bc5eb4d3285fb266f9dd5fb012d0b8`. This is a local digest of the official TLS download, not an independently signed vendor checksum.
- Expanded: 1,506 entries, 952,975,283 bytes. Every archive destination was resolved and checked inside the fresh workspace SDK directory before extraction; absolute/device/escaping paths and symlink entries were rejected.
- SDK root: `C:/Users/jntay/Documents/ChatGPT/My Model/colibri-strata/outputs-native/native-sdk-discovery/sdk-2026.4.1/openvino_genai_windows_2026.4.1.0_x86_64`.
- Installed runtime C API reports `2026.4.1-22982-e213a147257-releases/2026/4`.

| Artifact | Path under SDK root |
|---|---|
| Public C headers | `runtime/include/openvino/c` and `runtime/include/openvino/genai/c` |
| Release C/C++ vendor DLLs and device plugins | `runtime/bin/intel64/Release` |
| Required TBB dependency DLLs | `runtime/3rdparty/tbb/bin` |
| MSVC Release import libraries | `runtime/lib/intel64/Release` |
| CMake package files | `runtime/cmake` |

The SDK contains `openvino_genai_c.dll` (178,448 bytes), `openvino_genai_c.lib`, `openvino_genai.dll`, `openvino_c.dll`, `openvino_c.lib`, `openvino.dll`, `openvino_tokenizers.dll`, and CPU/GPU/NPU plugins. `dumpbin /exports` verified the real LLM and Whisper C pipeline symbols. The previous pip GenAI distribution omitted the separate C DLL; the full matching native SDK resolves that dependency. Use its coherent runtime and headers rather than mixing a 2026.4.0 wrapper with pip DLLs.

The official build defines library `openvino_genai_c` and target `openvino::genai::c`, linking publicly to `openvino::runtime::c` and privately to vendor C++ GenAI. C packaging is deliberately excluded from Python wheel builds. [Pinned build definition](https://github.com/openvinotoolkit/openvino.genai/blob/ab01d1771e4adaa0e1e08e9d630def170572506c/src/c/CMakeLists.txt), [packaging condition](https://github.com/openvinotoolkit/openvino.genai/blob/ab01d1771e4adaa0e1e08e9d630def170572506c/src/CMakeLists.txt).

## Actual native compiler and proof

Visual Studio Community 2022 17.14.28 is installed with VC tools. Compiler: `C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe`; actual version **19.44.35224**, x64. Build environment: `VC/Auxiliary/Build/vcvars64.bat`.

An ignored discovery probe, `outputs-native/native-sdk-discovery/c_sdk_probe.c`, compiled successfully with `/TC /std:c17 /W4`, linked `openvino_genai_c.lib` and `openvino_c.lib`, and executed using only native vendor DLLs. It queried the runtime version and created/freed both LLM and Whisper configuration objects. Output:

```text
OpenVINO native C: 2026.4.1-22982-e213a147257-releases/2026/4 | OpenVINO Runtime
GenAI LLM and Whisper C ABI: owned config create/free passed
```

This proves C compilation/linkage and configuration ABI availability, not inference, hardware placement, quality or acceleration. The build recipe is saved as `outputs-native/native-sdk-discovery/build_probe.cmd`. Runtime DLL search must include both Release and TBB directories; adding only Release leaves `tbb12.dll` unavailable. Configure those paths inside the launched process, without changing the user's global environment. A C application can use scoped `AddDllDirectory`/`LoadLibraryExW`; no Python interpreter is needed at inference time. The SDK's setup script runs optional Python installation checks, so direct path configuration is preferable for a Python-free launch.

## Native C workload routes and ownership

**Text:** use the opaque C LLM pipeline and generation configuration, then bounded result extraction. The constructor's variadic property count counts individual key/value arguments: one property uses count 2. The text callback has `__cdecl` ABI on Windows and returns running=0, stop=1, cancel=2; its string is borrowed for the callback invocation. Result strings use a two-pass caller-owned buffer with size including NUL; result/config/pipeline objects have vendor free functions. [Pinned C header](https://github.com/openvinotoolkit/openvino.genai/blob/ab01d1771e4adaa0e1e08e9d630def170572506c/src/c/include/openvino/genai/c/llm_pipeline.h).

LLM result conversion returns exact text for one output sequence. Multiple sequences add score prefixes/newlines. Enforce one return sequence for the simple protocol. [Conversion implementation](https://github.com/openvinotoolkit/openvino.genai/blob/ab01d1771e4adaa0e1e08e9d630def170572506c/src/cpp/include/openvino/genai/llm_pipeline.hpp#L79-L109).

**Speech and English translation:** the opaque C Whisper pipeline accepts a `const float*` sample array and sample count. Use model-compatible normalized mono 16 kHz input; task `translate` selects English translation, while `transcribe` preserves the source language. Retrieve exact text by index with the count/size APIs, then free the vendor result. Setting language or task to NULL clears the optional value. [C pipeline](https://github.com/openvinotoolkit/openvino.genai/blob/ab01d1771e4adaa0e1e08e9d630def170572506c/src/c/include/openvino/genai/c/whisper_pipeline.h), [configuration implementation](https://github.com/openvinotoolkit/openvino.genai/blob/ab01d1771e4adaa0e1e08e9d630def170572506c/src/c/src/whisper_generation_config.cpp).

**Trained image/video upscaling:** use the existing exported SR graph through core C read/compile/request/tensor APIs; preserve tiling, padding, overlap, alpha and video timestamp/audio behavior in C-owned orchestration. Core compiled-model property lookup can retrieve `EXECUTION_DEVICES`; check the requested explicit device before/after inference. The generic core route needs no GenAI pipeline or C++ client source.

## Verified API gaps that implementation must handle

1. The pinned Whisper C `generate` function has no streaming/cancellation callback. Implement bounded native jobs/chunks and publication cancellation; do not free a running pipeline or claim an exposed interrupt API. Core infer-request cancellation exists separately for directly managed graph requests.
2. Whisper's generic-generation-config getter **copies** the base configuration. Mutating that returned object does not set Whisper's inherited max-new-token bound. The approved C-wrapper extension below sets the inherited field on the actual owned Whisper configuration. The backend takes the pipeline's owned configuration copy and uses this setter, preserving all model defaults without temporary files or source-asset writes. [Original copy implementation](https://github.com/openvinotoolkit/openvino.genai/blob/ab01d1771e4adaa0e1e08e9d630def170572506c/src/c/src/whisper_generation_config.cpp#L49-L61).
3. GenAI C 2026.4.1 exports no tokenizer API or direct apply-chat-template setter. Core C has string-tensor creation and can execute the existing tokenizer IR with the tokenizers extension, but exact limits require matching the text that GenAI actually tokenizes.
4. `GenerationConfig` defaults to applying the chat template. Its JSON constructor ignores `apply_chat_template`; only typed AnyMap updates read it. Adding false to a temporary JSON file therefore does not establish raw-text/token-count parity. Resolve exact native template/count behavior before claiming prompt/context ceilings. [Pinned loader](https://github.com/openvinotoolkit/openvino.genai/blob/ab01d1771e4adaa0e1e08e9d630def170572506c/src/cpp/src/generation_config.cpp).
5. GenAI's C pipeline API does not expose its compiled-model handle or an execution-device property query. An explicit constructor device is a request, not independent node-placement proof. Core SR can obtain such proof directly; LLM/Whisper placement qualification remains a separate native replay/diagnostics task.

No SDK dependency is now missing for the basic C client build. Remaining work is native workload integration, precise prompt/context handling, cancellation/publication policy and actual model/device qualification. CPU/GPU/NPU DLL presence does not establish a physical device or qualify TPU/AIPU execution.

## SDK C-wrapper extensions (approved dependency build)

The gaps above are resolved by rebuilding the official matching C wrapper with three additional exports. They access the original declared SDK source types and public methods; there is no opaque-object reinterpretation or layout shim. Colibri engine/backend clients stay C, and only the vendor dependency's existing C++ wrapper is rebuilt.

```c
ov_status_e ov_genai_generation_config_set_apply_chat_template(
    ov_genai_generation_config *config, bool value);
ov_status_e ov_genai_whisper_generation_config_set_max_new_tokens(
    ov_genai_whisper_generation_config *config, size_t value);
ov_status_e ov_genai_llm_pipeline_render_chat_template(
    const ov_genai_llm_pipeline *pipe,
    const char *const *roles, const char *const *contents, size_t count,
    char *output, size_t *output_size);
```

The renderer constructs the official `ChatHistory`, then calls the pipeline's public `get_tokenizer().apply_chat_template(history, true)`. It accepts 1..64 system/user/assistant messages with valid UTF-8 and at most 1 MiB aggregate content. It caps rendered bytes, including NUL, at 1 MiB. `output=NULL` queries the required size; a short buffer or failure publishes neither data nor size. It changes no conversation state. The backend renders the native model template once, then generates with `apply_chat_template=false` and counts that exact rendered prompt using the model's tokenizer IR. [Public chat history](https://github.com/openvinotoolkit/openvino.genai/blob/ab01d1771e4adaa0e1e08e9d630def170572506c/src/cpp/include/openvino/genai/chat_history.hpp), [public tokenizer method](https://github.com/openvinotoolkit/openvino.genai/blob/ab01d1771e4adaa0e1e08e9d630def170572506c/src/cpp/include/openvino/genai/tokenizer.hpp).

- Reproducible recipe: `c/tools/build_native_genai_c.ps1`; original source manifest pins all 18 C-wrapper files by Git blob hashes to the matching commit.
- With the official SDK extracted, build with `pwsh -File c/tools/build_native_genai_c.ps1 -FetchSource -OutputRoot outputs-native/native-sdk-discovery/patched-c-api-utf8-v3`. Fetching downloads only the pinned small source subtree into a new discovery source directory. Existing sources are verified and retained; choose a fresh `-OutputRoot` to repeat a build. The recorded v3 directory already exists on this PC and is preserved.
- Minimal patch: `c/tools/native_genai_c_apply_template.patch`; final seven-file unified patch SHA256 `d60ca4483d76434644a36d715211886b5784f9fcad9d8168c001707db849e3ab`. Independent `git apply --check` against the original pinned files passed after the UTF-8 constructor correction.
- Native dependency output: `outputs-native/native-sdk-discovery/patched-c-api-utf8-v3/openvino_genai_c.dll` and matching import library. Independently measured DLL SHA256 `aa92f706b365d1ac5cf1e1298aafbf6d80fbc5d843563572fbe90e722b9c9f46`. Earlier `patched-c-api` contains only the text setter; `patched-c-api-chat-v2` has all three exports but lacks explicit UTF-8 model-path construction.
- Parent's final v3 `dumpbin` comparison retained every one of the original153 C exports and added exactly the same three requested APIs.
- A pure-C MSVC C17 probe compiled with `/TC /std:c17 /W4 /WX`, linked against the patched headers/import library, and executed text setter NULL rejection and false/true calls, Whisper max-new-token 137 round-trip through an owned base copy, and renderer NULL rejection with buffer/size invariance. No model was loaded by that probe. Successful real-model rendering remains part of the parent integration replay.
- Native load order/search: patched C-wrapper directory, matching original SDK Release DLL directory, and its TBB directory. Source weights, model configuration assets and original SDK binaries remain unchanged.

Use the text setter with false for raw/already-rendered text requests, the native renderer for completion/chat message histories, and the Whisper setter on the pipeline's owned configuration copy. Native tokenizer IR counting must still match GenAI's special-token behavior, and actual prompt-count parity must be verified during native model replay. These extensions add neither a general tokenizer C API, a Whisper interrupt callback nor a GenAI placement-proof interface.

## UTF-8 path correction and review

The original Windows C wrapper passed narrow model paths into `std::filesystem::path`, which MSVC interprets using its filesystem code page. On this PC that code page is1252; UTF-8 encoded Japanese/accented paths therefore needed an explicit conversion. The reproducible recipe now replaces exactly one `std::filesystem::path(models_path)` with `std::filesystem::u8path(models_path)` in each pinned LLM/Whisper pipeline implementation. The C++17 UTF-8 constructor is a public standard API; it adds no export, private type access or Colibri C++ client code.

Independent source review found exactly seven changed files among the18 pinned originals, corresponding to the three approved API additions and the two constructor corrections. Both built constructor files contain exactly one explicit `u8path(models_path)` and no remaining narrow model-path construction. The source recipe refuses an unexpected/repeated anchor. The v3 DLL digest above matches the actual built artifact. Parent's completed v3 export audit preserves153 original exports plus the same three additions.

The Windows C process now obtains both command-line arguments and all three SDK path environment values as wide strings, then makes bounded owned UTF-8 copies before entering the unchanged inference routines. Partial argument/environment allocations and the wide command-line block are freed on every exit. POSIX still uses its existing byte environment path.

Parent-authored `outputs-native/native-final-unicode-replay.json` records genuine native text and speech from accented/Japanese model directories while both core/GenAI DLL environment paths also use accented/Japanese directories. Both produced text and closed cleanly. The related earlier CPU learned-SR replay established the core graph path. This reviewer read those artifacts and reviewed the path corrections, without repeating model inference. Verdict: **approved** for these focused encoding fixes. This evidence does not establish arbitrary SDK versions, other public GenAI pipeline types or new physical-device acceleration.

Final parent gate: all six production CPU/GPU routes passed again using the wide argv/environment entry and v3 dependency. Windows PNG7 and Linux native build/C3+client20+PNG7 passed. The v3 build manifest now records the final checked seven-file patch digest as its primary identity; it retains the earlier recording digest separately with explicit original-source/built-source reconstruction provenance. Publication/qualification curation remains parent-owned.
