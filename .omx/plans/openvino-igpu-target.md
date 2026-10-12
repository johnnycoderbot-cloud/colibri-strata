# OpenVINO physical iGPU target check

User explicitly targets iGPU and requested inspection of their OpenVINO project,
including external drives. Found the actual uncommitted checkout at
E:/My Documents/Needle/openvino-windows-llm; preserve all external project files.
Live project runtime OpenVINO2026.2.1 reports CPU/GPU and Intel UHD Graphics770
(integrated), driver32.0.101.7088. Its historical TinyLlama INT4 single-run
benchmark reports CPU14.373/GPU25.790 tokens/s; this is not a new paired fork gain.

Fresh Colibri native C camera replay exposed default-device alias rejection:
explicit GPU.0 is rejected when SDK enumeration lists the single device as GPU.
Correct only GPU.0/NPU.0 to their unindexed default aliases. Preserve explicit
other-index refusal and execution-device verification. Root owns this focused
native C repair, regression tests, rebuilt Windows SDK replay and evidence.
No delegation or new team is needed for this small repair.

Verify SDK-free full backend ownership/device contract on Windows and Linux,
ASan/UBSan/leaks on Linux, then real trained native C super-resolution on the
physical iGPU. Record exact model/binary/source identities and clean shutdown.
Mark iGPU as an explicit runtime target in the broader state. Mojo dispatch
wiring to OpenVINO remains separate; do not claim new split-policy acceleration.
