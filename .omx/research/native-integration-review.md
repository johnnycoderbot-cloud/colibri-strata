# Native graph production/install review

Date: 2026-10-11. Reviewer: `/root/architecture`. Scope: production factory/runtime changes, native C build/install closure, optional dependency inventories and graph CI. No production source edits by this reviewer. Backend/process/client internals have separate owners/reviews.

Verdict: **approved** for the frozen native production-wiring/build/install slice and the additional narrow PNG palette/scanline patches. All actionable findings below are resolved and the relevant final gates passed. Full hardware qualification/publication remains the parent's responsibility; no NPU/TPU/AIPU or macOS replay is inferred.

## Resolved first-pass findings

The following records the initial findings and subsequent corrections. The final source follow-up below supersedes the intermediate checks described here; no listed source correction remains outstanding.

1. **P1 — release/install closure omitted the required C executable.** The production `c/stream_models.py:13` factory now always constructs `NativeGraphAdapter`; `c/native_graph.py:60` requires `native_graph(.exe)` beside the installed modules. The initial `make install` copied that binary only when already built, and `.github/workflows/release.yml:98` / `:161` build/copy inventories omitted it. The installed module closure test could pass while every graph workload failed before startup. Parent has changed Make's install prerequisite/copy to unconditional `native_graph`; release build/copy and shipped dependency-build assets still require verification.

2. **P2 — stale C engine/dependencies could survive source/header or flag changes.** Initial `native_graph` recipe compiled four C translation units in one command and was excluded from derived `ENGINE_RULES` because no `native_graph.c` exists. The actual generated `c/native_graph.d` contained only `native_graph: native_media.c native_media.h`; JSON/digest/backend dependencies were lost. `tools/clean.py` omitted the engine. Parent has split the four translation units into objects and added binary cleanup. Remaining checks: clean all four objects, force regeneration when an object's `.d` is absent, and fix the two-translation-unit test recipes for media/manifest. Use the repository's automatic dependency mechanism and regression gate; extending handwritten header lists does not resolve this class of bug.

3. **P2 — Windows CI did not compile the new production C code.** `.github/workflows/graph-runtime.yml:32` initially gated C builds/tests to Linux, and never invoked `test_native_graph_process.py`. Windows Python mock tests cannot catch MSVC C17 or real subprocess/PNG admission regressions. Run the SDK-free process test on both Windows/Linux, with the native builder, and keep fake SDK tests identified as contract tests. The former `test_stream_models.py` now imports the offline Python reference adapter; its CPU IR fixture is not production-native evidence and may skip without the Python reference extra.

4. **P2 — official SDK alone cannot satisfy the production C ABI.** Text/audio require three approved wrapper extensions that the stock vendor C DLL lacks. Install/release must carry a reproducible extension recipe/patch/source identity manifest, or explicitly require the source checkout for those dependencies. Current `.ps1` builder is Windows/MSVC; Linux engine source is portable but Linux wrapper reconstruction needs a recipe/qualification boundary. The SDK output path and version must stay coherent. No Python SDK fallback should be added to hide missing native components.

5. **Dependency guidance drift — resolved in current parent edits.** `requirements-runtime.txt` removed Python OpenVINO inference bindings while `pyproject.toml`'s runtime extra initially retained them. Parent moved those bindings to the explicitly offline `graph-reference` extra and aligned `runtime` with preparation/capture packages. The native route itself never imports that reference adapter.

## Verified production boundary

- `stream_models.make_adapter` imports only `native_graph.NativeGraphAdapter`; no Python model SDK execution or automatic reference fallback exists.
- `GraphEngine` still owns admission, sessions/queues and cancellation. It forwards the admitted memory budget to the native process factory through a private manifest copy.
- Production PNG validation delegates to the C validator when the adapter reports `inference_runtime=native-c`; WAV framing validation is standard-library-only. The model/tensor preparation is native. The legacy PNG validator branch remains for explicit injected test adapters, rather than a runtime inference fallback.
- Native build uses C17 source files and typed SDK C functions loaded at runtime. Windows uses MSVC `/TC`, a compiler response file, and normalized environment keys; Linux uses a C compiler and `dl`; the script excludes `-ldl` on Darwin. Actual macOS compilation/device execution has not been replayed.
- `GraphEngine._device_evidence` preserves `execution_devices_verified`; `executed_device` is null unless proof exists. Constructor selection is not silently promoted into execution proof. Physical NPU/TPU/AIPU qualification remains separate.

## Verification so far

Independent installed Python module closure test passed on Windows; the symlink alias test skipped because it requires Windows symlink privileges. This proves Python files are copied, not native executable/SDK closure.

An initial independent current-client run reported15 tests with8 failures. The client owner confirmed these were newly added red regressions before their implementation, covering result device substitution, failed force-kill ownership, failed QUIT cleanup, malformed/bloated validator metadata, empty READY backend and thread-start cleanup. This is explicitly pending until the owner freezes the corresponding fixes; it is not a qualification failure of a published artifact.

Follow-up after the client owner froze source: all20 client tests independently passed on Windows, including the genuine non-reading Python child used to fill its stdin pipe, deadline/cancellation, retained ownership on cleanup errors, READY/RESULT identity checks and bounded validator output. The owner reports the same20 tests passing in WSL. Initial red regressions above are now resolved.

Parent fixes are changing the source while this review runs. Final approval requires reviewing the frozen changes and recording the meaningful native dependency/process tests. Full real-model replay and documentation/publication remain parent gates, distinct from this source integration review.

## Follow-up source corrections reviewed

- Make install now builds and unconditionally installs `native_graph(.exe)`. Release engine build/copy inventories include it. Both install and release carry the Windows SDK wrapper recipe, patch and pinned source manifest.
- Each native graph C translation unit has its own object/dependency file, with the four objects in `AUTODEP_OBJS` and tied to `.build-config`. Missing `.d` files therefore trigger regeneration. Cleanup removes all four objects and the graph binary. Media/manifest test recipes link helper objects and compile only one translation unit each. Native backend link guards use target `IS_WIN`/Darwin rather than host OS.
- Windows CI now compiles the C17 executable and SDK-free contract tests using `build_native_graph.py --check`, and runs the real native PNG process test. Linux Make runs the same process test. Offline Python reference tests are named explicitly and use the separate graph-reference extra.
- Runtime extras no longer require Python model bindings. Native setup documentation explains executable build, stock-SDK missing exports, matching patched Windows SDK configuration and source-checkout dependency reconstruction. Linux/macOS wrapper rebuilding and real-model replay are explicitly unqualified, without implying that C cannot implement these workloads.
- Production device evidence remains honest: GenAI constructor selection has `execution_devices_verified:false` / `executed_device:null`; learned SR checks its actual execution-device property. No new acceleration/power claim is made by the graph port.

Parent-authored `outputs-native/native-final-production-replay.json` was read: six authenticated API replays (three workloads on CPU/GPU) report HTTP200, `inference_runtime=native-c`, and no loaded Python ML modules under the import guard. This is reviewed parent evidence, not an independent hardware replay by this reviewer. The artifact itself labels the runs as functional checks without throughput/power claims.

The final graph CI path filters also include `pyproject.toml` and the release workflow, resolving the minor maintenance recommendation so packaging/dependency-only edits trigger this gate.

## Additional PNG patch review

Parent requested a narrow independent review of the local STB palette guard and native main's exact scanline budget. No blocking finding:

- `c/third_party/stb_image.h:4961` scans the decoded one-byte palette-index plane before reading palette entries or allocating the expanded palette output. Indexes at/above the declared initialized palette length fail. STB's `stbi__do_png` cleanup frees the retained raw image/expanded/IDAT buffers when this function returns failure.
- `c/native_graph_main.c:39` verifies chunk framing, CRC, one IHDR, exact terminal IEND, 8-bit supported color types, bounded dimensions and source-pixel count at most `max_output_pixels/16` before allocating pixels.
- `:65-78` derives an exact row/filter-byte total for noninterlaced and all seven Adam7 passes, concatenates the bounded IDAT stream into a source-sized buffer, and checks it using a non-growing inflater buffer before STB's normal decoder. All compressed and expected byte counts remain within signed-int casts under the process's 32MiB compressed/16MiB output-pixel caps. Unknown critical chunks (including CgBI) are refused, so a headerless/alternate inflation path cannot bypass this same-stream check.
- Success alone publishes pixels/geometry. This review is specifically the palette/scanline change and its admission preconditions; it is not a replacement for broader process/manifest reviews.

Reviewed patch identities: STB image header `0cf24a63093060ce3bce350d89a57d08028ff6330705d1cf74eee361688efebc`; native main `51416e1caced8734097999b6144afa2cf1ac0fe37fd0ef5fa3026b0e9eafa10f`.

Parent final-gate update: Linux generated-dependency regression passed all8 tests. The three native C contract programs passed on Windows MSVC (225 manifest checks) and Linux GCC (233). Seven real PNG process tests passed on Windows and under Linux ASan/UBSan. Native/reference SISR pixel and alpha comparisons matched exactly for17x13,1x9 and450x3 (including a tile seam); native text prompt count22 matched SDK22 and generated text matched. Native GPU English translation closed cleanly for an English audio fixture; this does not establish translation quality on another source language.

The combined-gate huge-validator-output fixture initially had a too-short scheduling budget: it sometimes reached its bounded deadline first and raised `TimeoutError` instead of the expected malformed-output `ValueError`. The owner fixed only that test fixture by reading bounded4096-byte chunks and allowing a1s fixture deadline; dedicated hang/full-pipe cancellation tests retain50ms. Production bounds were unchanged. The owner reports the complete WSL `make native-graph-check` passing C3+client20+PNG7. After that final freeze, this reviewer independently reran all20 Windows client tests: pass, no skip,2.361s.

Parent also reports native GPU CLI text, trained three-frame video retaining one audio stream and source timestamps, plus explicit refusal of unavailable NPU, missing core library and the original wrapper lacking required exports. These establish particular functional routes/refusal behavior and do not measure acceleration. In the reviewed real-model artifact, no forbidden Python model package was loaded.

## Final source identities

- `c/stream_models.py`: `484c03e27a7b575cff1a9702efefd490eefe840cfc9253c7a3079b4efc936105`
- `c/stream_runtime.py`: `59c967c688b6ffc3adb633e3a2fe2b3d4428bfb326b2f2d8e78022c27a31c3cc`
- `c/native_graph.py`: `f19c5cc8ddf33c631a4f9043c05fbb06055feb92046b568c93c69f4939b2ef4c`
- `c/tools/build_native_graph.py`: `417b72d23d295baa349effe34ce3ddbd1f1ec43238602cda6457a5c7a6b705f5`
- `c/Makefile`: `2c6f0221efc33e939e44918c13033f2190589871adc1c19117fd8cc47388b3c4`
- `.github/workflows/graph-runtime.yml`: `d6638edde7d5d2ba7fd8032057f28eee7e6e6870b40c1956dcba3f52c333fc1d`
- `.github/workflows/release.yml`: `36f9867c9d8cb10a3935e2eb1900e50868b7fd30a57126632e6d80a0304f2cdc`

The native C workload ports are reviewed here as functioning implementations. Vendor internals remain native SDK dependencies behind their C ABI; Linux/macOS wrapper rebuilding is explicitly unqualified. GenAI constructor selection remains distinct from SR's property-based device proof. Broader checkpoint/family, precision and physical accelerator claims require their own evidence.

## Windows Unicode follow-up review

The new wide command-line entry itself is sound: `CommandLineToArgvW` has correct Shell32 linkage, argc1..64 is checked before allocating the NULL-terminated argv table, each UTF-8 conversion is bounded to16384 bytes including NUL and rejects invalid UTF-16, and every partial argument allocation plus the wide argv block is freed on all exits. Only the entry representation changes; the existing `graph_main` engine keeps the same behavior. The compiler helper's UTF-8 BOM response file preserves Unicode paths and uses `/TC`; its legacy-console-safe status print cannot turn a successful build into a UnicodeEncodeError. The Apple feature macro correction matches the already-reviewed backend and adds no macOS execution claim.

Two related path boundaries were sent to the owner for qualification/correction before making a broad Unicode claim:

- `graph_main` initially still reads the three SDK library paths through narrow `getenv`, while the backend interprets these bytes as UTF-8. This host's Windows active code page is1252. Windows should obtain those environment values as wide strings and convert them into owned UTF-8, or pass the paths through the now-correct wide argv path.
- The existing patched official GenAI wrapper constructs LLM/Whisper paths with `std::filesystem::path(models_path)` (its pinned `llm_pipeline.cpp:102` / `whisper_pipeline.cpp:279`). MSVC's char-path conversion uses `__std_fs_code_page()`, while `std::filesystem::u8path` explicitly uses UTF-8. A Unicode core-SR replay does not establish Unicode GenAI model-path behavior. These are original SDK path semantics, not an unsafe object-layout shim; explicit UTF-8 construction or a stated limitation needs separate qualification.

New entry/builder source read was successful; the parent reports native CPU SR replay from an accented/Japanese pack directory and seven PNG tests compiling/executing from a Unicode temporary directory. No independent model run was performed in this narrow follow-up.

Narrow follow-up verdict: **approved** for the wide argv entry, bounds/cleanup, Shell32 linkage, BOM compiler response file, console-safe status output and Apple feature macro. The existing environment/GenAI-path encoding findings above are separate boundaries and must not be treated as qualified by this entry fix.

Follow-up identities (superseding earlier identities for these files):

- `c/native_graph_main.c`: `0f490134893e5615775bb14646bfc47ab9ae8f307c4195b0e64d60d9e07e0ea2`
- `c/tools/build_native_graph.py`: `5358a55222ad7e507231395b6efa6d01d2db1d77b67459ae53a230d67ad0e139`
- `c/Makefile`: `bf39e5e62f4a7f50b1bb235568750416606844daf937ebb71d61ba56cada67fe`
- `c/native_graph_manifest.c`: `f763b7f7c9e7b9b56dbc420371286941b41e5c8b69fa5d55919bba7b925d4523`
- `c/tests/test_native_graph_process.py`: `3cf3b839af7567c9063cbcc09aabacd2e9d63540982f5b49c2a503ec7b6a10bf`

## Unicode findings resolved

Both related encoding findings above are now **resolved and approved**. Windows entry obtains all three SDK library environment values through `_wgetenv`, converts each to an owned UTF-8 buffer with the same16384-byte including-NUL bound, passes them explicitly into `graph_main`, and frees all three on every exit. Missing values remain NULL; command-line library overrides still work. Partial allocation failures retain no buffer. POSIX uses its original `getenv` path.

The matching official SDK recipe uses public C++17 `std::filesystem::u8path(models_path)` in the LLM and Whisper constructor. Exact-one-anchor guards reject source drift. Independent comparison of the18 pinned originals to built source found exactly the expected seven changed files; `git apply --check` passed the reconstructed seven-file patch. Built v3 DLL SHA256 matches `aa92f706b365d1ac5cf1e1298aafbf6d80fbc5d843563572fbe90e722b9c9f46`. These constructor changes add no C ABI export or object-layout access.

Parent `outputs-native/native-final-unicode-replay.json` was read: native text and speech both use accented/Japanese model directories and core/GenAI DLL environment paths, produce actual text and close cleanly. This is focused parent replay evidence; no model was rerun by this reviewer. It closes the previously stated core-SR-versus-GenAI Unicode qualification gap for the recorded models/platform.

Latest reviewed identities superseding the corresponding entries above:

- `c/native_graph_main.c`: `8f1495895833b893882cf8a14428a88a07eef619b4365584345780db2b57e062`
- `c/tools/build_native_genai_c.ps1`: `aa4e021dbefda2135216ee9a4ce5cf155cf06811d753ac455525f644f0f50c7f`
- `c/tools/native_genai_c_apply_template.patch`: `d60ca4483d76434644a36d715211886b5784f9fcad9d8168c001707db849e3ab`

No unresolved blocking finding remains in the reviewed production integration or focused Unicode corrections. The parent owns the final v3 export audit and six-route replay; previously stated broader device/platform/precision limits remain unchanged.
