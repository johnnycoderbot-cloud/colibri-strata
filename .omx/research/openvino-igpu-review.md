# OpenVINO default-index alias review

Date: 2026-10-12. Reviewer: `/root/architecture`. Verdict: **approved**.

No blocking finding in the scoped repair. Production source was read-only; this review added only this artifact. No compiler, inference, benchmark, external-project modification or new agent was started by the reviewer.

## Source contract

`c/native_graph_backend.c:140-147` preserves exact matches and adds only `GPU.0` -> `GPU` and `NPU.0` -> `NPU`. It does not erase or round other indices, accept CPU substitution, or introduce AUTO/MULTI/HETERO routing. The existing unindexed request behavior is unchanged. Device admission at`:366` uses that same predicate while still passing the caller's explicit device string to the SDK; a failed compile remains a failure rather than a fallback.

For the camera graph, `cng_placement` at`:277-295` still parses a bounded single execution device, checks device syntax and matches it against the requested device before setting `execution_devices_verified`. It is called on compilation and after inference (`:343`, `:465`). Other GPU indices and CPU substitution remain rejected for an explicit zero-index request. Text/Whisper pipeline placement is not newly established by this camera-graph repair.

The tests at `c/tests/test_native_graph_backend.c:160-165` exercise unindexed GPU/NPU inventory, default-index acceptance, nonzero-index refusal, and GPU request rejection when the compiled property reports CPU or GPU.1. Successful and failed cases call the existing ownership check for zero live fake handles/libraries. NPU cases are SDK stubs; they do not qualify a physical NPU workload. The new alias is used directly during inventory admission; real camera execution additionally retains its compiled-property proof.

## Evidence checked

Read `.omx/research/openvino-igpu-project-check.json` and its producer `outputs-native/check_openvino_igpu_target.py`. The producer creates a genuine native C camera adapter, checks native runtime/device/compiled-device fields, compares the GPU and GPU.0 PNG hashes, verifies clean closure and requires the concrete unavailable-GPU.1 diagnostic. Its control process reports no loaded Python ML modules. The evidence's native backend hash matches the reviewed source.

The external checkout `E:/My Documents/Needle/openvino-windows-llm` has live OpenVINO2026.2.1 inventory CPU/GPU and Intel UHD Graphics770 marked integrated, with recorded Windows driver32.0.101.7088. The two fresh equivalent Colibri GPU/GPU.0 camera replays use the separate matching native OpenVINO2026.4.1.0 SDK's C core APIs and trained `single-image-super-resolution-1032` model. The native output is 512x384, selected/compiled device GPU.0, with `execution_devices_verified=true`; GPU and GPU.0 output hashes match and GPU.1 refuses. This establishes the fresh native C camera route on the physical iGPU, not a new paired throughput comparison. The external historical TinyLlama one-run CPU/GPU numbers remain historical and are not fork acceleration evidence; their completion-token counts are project-reported rather than newly checked SDK-native counts.

Windows MSVC build/full fake-SDK contracts and Linux ASan/UBSan/leak gates are recorded as passed by the owner. The reviewer inspected their source coverage and recorded replay; those jobs were not independently repeated. Mojo-to-OpenVINO coordinator wiring, current text/Whisper placement proof, physical NPU qualification and new speedup remain outside this repair.

## Reviewed identities

- `c/native_graph_backend.c`: `5192a6260f55514c1179906f6f76dd1e2584346c4b9a733aad7959ad9868e3b7`
- `c/tests/test_native_graph_backend.c`: `2af883dfa30ecd4caa5ccf7acdbdb78427aed3ba79536360cd8e1e3d04229ac6`
- `.omx/research/openvino-igpu-project-check.json` (frozen metadata finalizer): `55f4ed7414033676fd9595a4666db049ff1c379c66e9fdfec51e37a646865fff`
- `.omx/plans/openvino-igpu-target.md`: `17216b745acb7d4dee3ba8908cca6a8fca3c7ff18f2a10d83db19dd1efdd5f34`
- Native replay binary (recorded evidence): `708a3884a2459c82fa4128f4c26cb7df4b25d9c0332809d14a034ab580b330b1`

Metadata-only follow-up reviewed: driver/native SDK labels, the two-replay scope and historical token-count caveat are consistent with the existing evidence. Native backend/binary/input/output hashes and parity/closure/refusal fields are unchanged. Verdict remains **approved**. No test or inference rerun was performed for this finalizer.

Any later source change or regenerated evidence requires a refreshed recording.
