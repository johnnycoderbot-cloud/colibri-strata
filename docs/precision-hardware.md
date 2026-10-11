# Precision and hardware validation

The portable head core supports **true signed INT4, INT8 and INT16 weight storage**.
INT16 here is not FP16/BF16. These are lossy quantized weights; target-model
verification remains mandatory. Real model-specific trained weights and live
Colibri engine adapters from the original roadmap are still pending.

## What is ready and tested

| Path | INT4 | INT8 | INT16 | Evidence / limits |
|---|---|---|---|---|
| CSQ1 portable codec | Tested | Tested | Tested | Canonical bytes, bounds, error limits, malformed data, cross-language byte equality |
| Parallel quantized heads, CPU | Tested | Tested | Tested | Scalar/OpenMP, mixed precisions, transactional errors; dequantized F32 activations with double reference accumulation |
| ONNX head graph, CPU | Tested | Tested | Tested | Static graph, exact target identity, finite/shape checks and actual provider node profiles |
| Intel UHD Graphics 770, DirectML | Tested | Tested | Tested | Real physical GPU; integer weights decoded to F32 graph; exclusive DirectML compute profiles and reference parity |
| CUDA/TensorRT, other GPU providers | Adapter | Adapter | Adapter | Explicit ONNX provider route; absent provider/refused operations fail rather than silently substituting CPU; not physically tested here |
| Intel/Qualcomm/other NPU providers | Adapter | Adapter | Adapter | Explicit OpenVINO/QNN/etc provider options; provider availability alone does not certify a physical NPU or native integer arithmetic |
| Axelera Metis M.2 110m, SDK 1.8.0 | Head compilation pending | Native vendor artifact tested | Head compilation pending | 1,346/1,346 ResNet50 golden frames on actual Metis, not a Colibri head-bank test |
| Google TPU via JAX | Adapter | Adapter | Adapter | Explicit device placement and completion checks; CPU smoke and absent-TPU refusal tested; no physical TPU available |
| Coral Edge TPU | Pending | Pending | Pending | Requires a separate static fully quantized eight-bit TFLite/compiler adapter |
| Lexar NQ7A1 NVMe, C: filesystem | Tested | Tested | Tested | 64 MiB random regular-file reads checked against SHA256 and request identity; all three stored CSQ1 formats decoded; buffered I/O |

GPU/ONNX/JAX runners expand integer storage into F32 tensors. These are compatibility
paths, not native INT4/INT16 accelerator kernels or a speed guarantee. ONNX graph
export can be supplied to a vendor compiler; exporter success does not establish
that a compiler places all operations on an accelerator. JAX precision selection
does not make signed INT16 storage equivalent to TPU BF16 arithmetic.

## Serialization and compute contracts

`c/strata_quant.h` stores little-endian CSQ1 blocks with a 24-byte versioned header
and F32 maxabs per block. INT4 uses low nibble first and range [-7,7], INT8 [-127,127],
INT16 [-32767,32767]. Reserved most-negative codes are refused. Odd INT4 padding
must be zero; zero scales must be positive zero; nonzero blocks contain an extreme
code. Exact lengths, dimensions, finite scales and output aliases are checked.

`strata_quant_heads.h` supports different precisions per head and per projection.
All heads consume the same immutable hidden state. Logits/tokens publish only when
every head succeeds; failed arithmetic or a malformed final head cannot publish a
partial prediction. Scratch space can change on a failed call and must not alias
immutable weight/hidden memory.

## Commands

```sh
make -C c strata-check strata-check-scalar strata-sanitize
make -C c strata-fuzz STRATA_FUZZ_CASES=1000000 STRATA_FUZZ_SEED=20261010
make -C c strata-libfuzzer             # requires Clang; bounded 30-second campaign
python -m pip install -r experimental/strata/requirements-training.txt
python -m pip install -r experimental/strata/requirements-runtime.txt
make -C c strata-training-check strata-runtime-check

python experimental/strata/heads.py --trace target-trace.jsonl \
  --out target-heads.json --export-c target-heads-int4.h --weight-bits 4
```

Use `--weight-bits 8` or `16` for the other integer formats. The JSON training
pack retains F32 parameters; the generated C header embeds actual CSQ1 integer
bytes. `precision_smoke.py` checks exported bytes and predictions against C.

Optional standalone GPU/NPU runner:

```python
import json, sys
sys.path.insert(0, "experimental/strata")
from runtime import export_onnx, run_onnx, run_jax
pack = json.load(open("target-heads.json"))
model_bytes = export_onnx(pack, bits=4)  # opset17, IR8, static input [1,H]
result = run_onnx(pack, hidden, provider="DmlExecutionProvider", bits=4)
# Device options are vendor-specific; an unavailable NPU is an error.
result = run_onnx(pack, hidden, provider="OpenVINOExecutionProvider",
                  provider_options={"device_type":"NPU"}, bits=8)
# A TPU request never silently uses CPU.
result = run_jax(pack, hidden, platform="tpu", bits=8)
```

DirectML needs its optional Windows provider package in an isolated environment.
The documented local GPU test used onnxruntime-directml 1.24.4. ORT CPU 1.26.0,
ONNX 1.18.0, JAX CPU 0.6.1 and Torch 2.10.0+cpu were exercised. Each accelerator
request disables CPU fallback and checks node profiles; wrong provider proof is
an error. An NPU provider's own device routing also needs vendor-specific evidence.

`hardware_check.py --pack ... --provider ... --out report.json` checks standalone
heads plus bounded regular-file integrity. It creates and removes its own 8 MiB
fixture; it never writes a raw disk. This is buffered filesystem I/O, not O_DIRECT,
GPUDirect Storage, a durability certification, or a storage-throughput benchmark.

`axelera_check.py` accepts explicit existing SDK Python/runtime/device paths and
a precompiled model.json. It invokes the SDK with UTF-8 and a bounded timeout,
checks every golden frame, and propagates the actual SDK exit status. It does not
install/reset the device, flash firmware, or change power settings. On Windows,
Metis inference is native; model compilation runs on Linux/WSL. A head-bank ONNX
artifact still needs appropriate calibration, vendor compilation, placement
inspection and real head-output validation before use on Metis.

## Adversarial findings and verification

- 1,300,000 seeded cases passed under GCC ASan/UBSan: 1,000,000 with seed20261010
  and 100,000 each with 20261011, 0 and UINT64_MAX. Tests cover malformed/truncated
  codecs, odd nibbles, bounds, finite/overflow errors, canaries, mixed head shapes,
  failed output atomicity, canceled/stale prefetch tickets and causal rollback.
- Python and C initially differed on negative-zero scale and missing extreme-code
  rejection. Both now enforce the same canonical rules, with regressions.
- A file-tier completion callback raced a fresh submission after result(). Capacity
  is now released before publishing completion; active cancellations retain slots.
- A FIFO input blocked before the regular-file check. Reads now precheck file type,
  use POSIX nonblocking open, and verify the actual descriptor. A bounded FIFO test
  passes on Linux; Windows records that POSIX-only test as inapplicable.
- SDK statistics crashed under Windows cp1252. The local adapter forces UTF-8,
  handles both full-verification summary styles and preserves nonzero SDK status.
- Explicit accelerator CLI options require a head pack; they cannot silently turn
  into a passing storage-only check. Existing report/diagnostic files are preserved.
- The full upstream portable C suite exposed a CRLF checkout bug in the SSD fixture
  reader. The test now strips record line endings while preserving escaped payload
  control bytes. The complete suite passes after that correction.

The failure tests are evidence for these paths, not a claim of zero bugs. Production
engine adapters, real-model acceptance/quality, hardware DMA integration and native
integer kernels remain distinct gates.

Primary references: [DirectML provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html),
[Axelera compiler](https://docs.axelera.ai/sdk/reference/compiler/compiler-api/),
[Axelera ONNX constraints](https://docs.axelera.ai/sdk/reference/compiler/onnx-support/),
[Windows Metis](https://docs.axelera.ai/sdk/user-guides/windows-setup/),
[JAX TPU](https://docs.cloud.google.com/tpu/docs/run-calculation-jax),
[Coral requirements](https://www.coral.withgoogle.com/docs/edgetpu/models-intro/).
