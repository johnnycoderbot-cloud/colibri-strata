# Precision and hardware hardening

User request: signed INT4/INT8/INT16 readiness; adversarial bug/failure tests;
GPU, NPU, AIPU and NVMe capability. Clarification: user has AIPU and wants
standard NPU/GPU/TPU coverage.

Interpretation: INT16 means signed integer storage, distinct from FP16/BF16.
Implement and test portable codecs and independent quantized head execution;
add guarded real hardware/export paths and failure refusal. Do not present a
detected device, mock backend or CPU fallback as hardware-executed inference.

Slices / owners (native in-session, not detached OMX team):

1. /root/memory_primitives: new c/strata_quant.h + test_strata_quant.c; CSQ1
   INT4/8/16 codec, canonical serialization, reference matvec and malformed data.
2. /root/parallel_heads: new strata_quant_heads.h + test_strata_quant_heads.c;
   actual quantized bank inference, parallel/serial parity, atomic failed outputs.
3. /root/architecture: fuzz_strata_core.c and adversarial source/vendor audit;
   bounded mutations, canaries, state failures, independent review verdict.
4. /root: quantized training/export bridge, runtime hardware adapters and capability
   inventory, safe NVMe file tests, optional ONNX/provider and JAX/TPU paths,
   build/CI, published exact verification evidence.

Verify: strict GCC serial/OpenMP, ASan/UBSan, seeded structured mutation campaign,
INT4/8/16 C/Python export parity, provider error/fallback tests, real file I/O
integrity/short-read tests. Run existing upstream quant/backend/store tests as
relevant; GPU/NPU physical execution must be explicitly identified. AIPU needs
qualified compiler output and actual SDK/runtime; no automatic driver/firmware changes.

Exit: portable precision tests and failure cases pass; hardware/export capability
is recorded precisely; discovered bugs fixed; changes published with truthful
physical-versus-simulated evidence. Real-model trained head quality and all-family
live engine integration from the prior roadmap remain separate pending work.
