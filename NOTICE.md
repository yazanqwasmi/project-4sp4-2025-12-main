# Attribution and Provenance

This project was built for **COMPENG 4SP4 — High-Performance Programming** at
McMaster University (Fall 2025). It is published as a portfolio artifact.

## What is course-provided vs. authored

The course distributed a skeleton: API headers, benchmark harness scaffolding,
build system, and empty kernel stubs. Every source file therefore carries a
`SwiftWare Lab` copyright header from that skeleton.

**Course-provided (SwiftWare Lab):**

- Public API signatures in `include/` — `gemm.h`, `gemv.h`, `spmm.h`,
  `spmv.h`, `dense_nn.h`, `sparse_nn.h`, `def.h` (the `DenseMatrix`, `CSR`,
  and `ScheduleParams` types, marked "please do not change")
- Benchmark harness skeletons and the SLURM job script
- CMake project scaffolding

**Authored by Yazan Qwasmi:**

- All CPU kernel implementations — `src/gemm.cpp`, `src/gemv.cpp`,
  `src/spmm.cpp`, `src/spmv.cpp`
- Both CPU network implementations — `src/dense_nn.cpp`, `src/sparse_nn.cpp`
- All CUDA kernels — `include/kernels.cuh`, `src/gpu_dense_nn.cu`,
  `src/gpu_sparse_nn.cu`
- The full unit-test suite in `test/` (26 tests across 6 suites)
- Python tooling — `script/dense_nn.py`, `script/sparsify_weight.py`,
  `script/plot.py`
- Portability, CI, and packaging work described in the README

## Licensing

The `SwiftWare Lab` headers restrict redistribution of the course skeleton.
This repository is shared for the purpose of demonstrating the authored work
above. No open-source license is granted over the course-provided material,
and none is claimed. If you are course staff and would like this repository
taken down or made private, contact me and I will do so immediately.

## A note to current students

If you are taking 4SP4, submitting any of this as your own is an academic
integrity violation. Read it if it helps you understand tiling or CSR; do not
copy it.
