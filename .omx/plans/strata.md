# Implementation slices

1. Architecture map (native worker architecture / explorer): inspect all upstream
   families, speculation rollback and actual Strata mechanisms. Exit: cited matrix.
2. Parallel head core (native worker parallel_heads / executor): c/strata_heads.h,
   behavior tests and microbenchmark. Exit: independent heads, checked dimensions,
   transactional target-only acceptance, serial/OpenMP agreement and rollback tests.
3. Memory core (native worker memory_primitives / executor): c/strata_memory.h and
   behavior tests. Exit: INT8 error/storage evidence, bounded deterministic hot-set,
   two buffers that cannot be reused ahead of backend event fences.
4. Parent / executor: offline head trainer/export, build target, fork README,
   integration and benchmark protocol, read-only GitHub CI. Exit: training smoke,
   registry coverage and verification evidence.
5. Parent / reviewer: audit added code, compile both thread modes, sanitizer tests,
   upstream speculation regression, publish fork and tracked follow-up tasks.

Runtime is native in-session workers, not detached OMX/tmux. No mock worker claims.
Engine adapters and hardware quality/performance are separate pending tasks and
must never be reported complete by the bootstrap checks.
