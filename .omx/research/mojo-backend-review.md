# Native Mojo/MAX C backend independent review

Date: 2026-10-11. Reviewer: `/root/architecture`. Scope: `native_mojo_backend.h/.c`, native tensor/file CLI, launcher/build/resource configuration, fixture qualification and hardware claims. Production files were read-only; this reviewer owns only this report.

Verdict: **approved** for the native compatible-graph backend and operator test path. No unresolved blocking source finding remains. This is not a full Colibri model/decoder adapter or universal accelerator qualification.

## Finding resolved

The original geometry expression `(MAX_BYTES*8-7)/bits` rejected a byte-aligned tensor at exactly128MiB by one element. Owner changed the bound to `MAX_BYTES*8/bits` before the bounded ceiling-byte calculation and added the allocation-free public `cmj_tensor_bytes` admission query. Tests exercise exact128MiB and one-over for all nine dtypes, preserving the caller's result on failure. No128MiB allocation is needed for these boundary tests. Integer widths and the cap keep the later multiply/add within size_t on supported hosts.

## ABI and ownership review

Reviewed the actual installed **MAX26.6.0** C headers under `/tmp/colibri-mojo-sdk/lib/python3.12/site-packages/max/include/max/c`, rather than reconstructing an opaque layout. Native sources include the official headers. The official [MAX C API](https://max.modular.com/api/c/) documents inference and asynchronous ownership boundaries; installed headers are the pinned ABI reference.

- `M_compileModelSync` consumes the compile-config pointer and resets it to NULL even on error (`model.h:99-123`). Cleanup correctly tolerates that consumed handle. Runtime/compile/status handles are otherwise released by their matching SDK functions.
- Tensor specs copy names; `M_borrowTensorInto` copies the spec and borrows input data (`tensor.h:27-49`, `:103-123`). The header expressly supports staging a host pointer to a requested accelerator. Colibri shadow-copies caller data before SDK borrowing and keeps those shadows until synchronization and map destruction.
- Synchronous execution returns resolved output tensors (`model.h:175-197`). The backend synchronizes the requested device, obtains an owned tensor wrapper and device handle, checks source output placement, copies through the official host-copy API and synchronizes both source/host before dereferencing host data (`native_mojo_backend.c`, run/output path).
- `M_getTensorByNameFrom` returns an owned wrapper borrowing a map tensor; its wrapper is freed before the output map. Device handles, specs and host copies are independently freed by their matching APIs (`tensor.h:125-139`, `:163-228`). Output data is copied into C-owned memory before SDK references are released.
- Safetensor registry data is borrowed; its archive must outlive both registry and initialized model (`safetensors.h:134-159`). Engine destruction frees model first, then registry/archive, compiled model/context and devices. Archive/registry remain live through initialization and all calls.
- All result structures are staged separately and copied to caller outputs only after every requested output is validated/copied. Error paths free staged allocations and leave caller output structures unchanged. Occupied/duplicate output names are refused. The caller externally serializes an engine and does not destroy it while a call runs.

Header SHA256 identities used in the independent ABI review:

| Header | SHA256 |
|---|---|
| tensor.h | `075b25d3f5a666026e8fa049587040770b766bae300f5666189bfabbb62560b7` |
| model.h | `76548bb8a175e4460125f292ce2b4f510df7924b2830bb212520bbea4375ed65` |
| device.h | `87d831971117a6a876404b650fac52fbe523a325f3b8e27094021479766014c1` |
| context.h | `afd9317011df1fe917c2f5bd2bed21aeb7adbaa0f202b9f6fb2095d59b6117c6` |
| weights.h | `8c1a61d2d45b1bd65e36a19a58b1f8728a25e61cc323f5b694019468df29b50b` |
| safetensors.h | `7160494c86fb59470444ca8425df4341b18434140c4b0774d9869e18136f0c9b` |

## Data, device and process contracts

Inputs/outputs are bounded to32 tensors, rank8 and128MiB aggregate storage/result budgets. Positive dimensions, multiplication overflow, declared byte count, dtype and floating finite values are checked. Packed signed INT4 is low-nibble-first and expands to signed INT8; graph input declares INT8. Native packed INT4 output is refused. The integer fixtures compute in INT64. U8 is explicitly unsigned; signed4/8/16 semantics are separate from floating F16/BF16.

CPU is an explicit host selection; accelerator IDs are exact, checked against the runtime count. The adapter performs no CPU retry. Output source placement must match that selection, and host copies must identify host0 before data access. The proof covers output tensor location, not every graph node. The public device header's accelerator enumeration concerns runtime GPU devices; it cannot establish physical NPU/TPU/AIPU support or classify an unknown device.

The C CLI reads regular, bounded, non-symlink final files through descriptor identity/size/time checks; rejects raw/decoded NUL and duplicate/nonfinite JSON; resolves relative input files beside the manifest; and publishes output files exclusively in a new directory. On publication failure partial files may remain and exit0 is withheld, as documented. C API result staging is transactional; disk publication is explicitly not described as atomic.

The builder obtains actual SDK paths from pinned package metadata, generates a matching `modular.cfg`, embeds the resource directory and native library rpath, and does not import SDK Python model bindings into production. An explicit config override is supported; the launcher invokes an absolute executable path so installed adjacent `.sdk` lookup works. This resource file contains host-specific absolute paths: documentation correctly requires rebuilding on the destination host.

Timing now includes input reads, compile/load, runs with shadow/host copies, publication, and model/output/input/JSON cleanup before the cleaned timestamp. Output metadata names/shapes are owned copies and remain valid after input JSON is freed; pointers are cleared before common cleanup. External process wall time additionally includes loader/startup. These are synthetic graph timings, not LLM tokens/s or a Colibri speedup result.

## Verification evidence

Independent launcher run: all4 `test_mojo_cli.py` tests passed on Windows without loading SDK bindings. They check exact shell-free device arguments, invalid device/budgets/Windows refusal, timeout/native error with no retry and blocked Python model imports.

Parent's final native installed-SDK contract suite reports **65 checks passing under ASan/UBSan with leak detection**, including the all-dtype boundary tests, model-incompatible valid shapes, same-byte wrong dtype, occupied output, nonfinite input, duplicate/missing outputs, limits and ownership. Parent also reports final external-weight CLI sanitizer/leak success. These are reviewed owner evidence, not additional runs by this reviewer.

Read `outputs-native/mojo-final-reviewed-replay/report.json`: F32 projection/ReLU, signed4/8/16 and external-weight fixtures all match their reference bytes. The probe reports SDK26.6.0, CPU available, accelerator_count0 and WSL=true. ACCELERATOR.0 refusal is explicit and publishes no output directory/result. Final replay tools require exact reference bytes and identify the run as functional synthetic evidence. Existing native/OpenVINO model formats are not silently treated as MAX graphs.

## Qualification limits

Local execution is Intel CPU in Kali WSL, not native bare-metal OS execution. Operator commands are ready for supported native hosts; actual GPU/NPU/TPU/AIPU execution remains unqualified. Official [Mojo requirements](https://mojolang.org/docs/requirements/) and [Modular packages](https://max.modular.com/packages/) require supported architecture/OS/drivers; native Windows is explicitly refused. Hardware visibility does not establish compatible model operators or a compiler target.

Native Colibri tokenizer/decode/workload adapters, model-state handling and SSD expert paging into this backend remain separate integration slices. Tensor budgets are not a model/compiler/weight memory cap. No accelerator speedup, whole-node placement, native INT4 math or universal-device discovery is claimed.

Reviewed production identities:

- `c/native_mojo_backend.c`: `eeb478aba8bfbb7fa03862d7471892bd901468c1650505761f69c4de1fe814a7`
- `c/native_mojo_backend.h`: `f812bfeb1e313231443f0f932d4bd95387fdb335f63efa00936843d34091ddaf`
- `c/mojo_runtime_main.c`: `4d2eb3d2d167c9627e0541a40a6f5e3cbc4c954b4e6e074a3ab6ba0bc2467629`
- `c/tools/build_native_mojo.py`: `85c70d045cba8b30a5890d83ed5a49e38162c9582df55e67c6592341265af60b`

The `.omx` Mojo state is still in progress while the owner curates/publishes qualification. This approval closes the independent review slice; broader hardware and decoder integration remain explicit.
