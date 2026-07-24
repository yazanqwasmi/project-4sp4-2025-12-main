# Sparse DNN Inference: CPU and GPU Kernel Engineering

[![CI](https://github.com/yazanqwasmi/sparse-dnn-hpc/actions/workflows/ci.yml/badge.svg)](https://github.com/yazanqwasmi/sparse-dnn-hpc/actions/workflows/ci.yml)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![CUDA](https://img.shields.io/badge/CUDA-12-76b900)

Hand-written dense and sparse linear-algebra kernels for MNIST inference,
benchmarked against Intel MKL, cuBLAS, and cuSPARSE.

The network is a 784 → 512 → 10 MLP (`tanh`, then `sigmoid`, then `argmax`).
Nothing is delegated to a math library: GEMM, GEMV, SpMM, and SpMV are all
implemented from scratch for both CPU and GPU, then measured against the vendor
implementations of the same operations.

The interesting result is where the custom kernels win and where they lose.
A tuned register-tiled GEMM still reaches only 29–62% of cuBLAS throughput,
because cuBLAS is very hard to beat at dense work. The custom CSR SpMM beats
cuSPARSE by 1.23–1.46× across every batch size tested, because a kernel
specialized to one sparsity pattern can skip the generality cuSPARSE must pay
for.

---

## Headline results

Measured on the McMaster ECE cluster: 20-core Intel CPU, NVIDIA RTX Ada 2000
GPU, single node under SLURM.

| Comparison | Result |
|---|---|
| Custom CSR SpMM vs. **cuSPARSE** | **1.23× – 1.46× faster**, widening with batch size |
| Custom GEMV vs. **cuBLAS** (256×256) | **1.05× faster** |
| Custom GEMV vs. **cuBLAS** (larger) | 0.88× – 0.91× |
| Custom GEMM vs. **cuBLAS** | 0.29× – 0.62× |
| Sparse vs. dense cuBLAS, batch 10–100 | **~2× faster** |
| Sparse vs. dense cuBLAS, batch ≥1000 | 0.31× – 0.57× |
| Dense baseline accuracy (MNIST) | ~92% |

**Accuracy vs. sparsity.** Magnitude-based pruning holds accuracy near the ~92%
dense baseline up to roughly 75–80% sparsity. Past 85% it falls off sharply,
eventually below 50%, as pruning starts removing genuinely informative weights.
Runtime, meanwhile, improves monotonically with sparsity. The usable operating
point is therefore around 75–80%: most of the speedup, almost none of the
accuracy loss.

**CPU kernel ordering.** Runtime follows `GEMM ≫ GEMV > SpMM ≫ SpMV`, matching
the arithmetic intensity of each: GEMM does O(M·N·K) work while GEMV streams
linearly at O(M·K), and the sparse variants touch only stored non-zeros.

> **Note on figures.** The plots these numbers came from were generated on the
> cluster and are not reproducible off it (they need the course MNIST data and
> the ECE node). `script/plot.py` regenerates every figure from the benchmark
> JSON if you have cluster access.

---

## Build and run

The CPU path builds and tests anywhere, with no MKL, no CUDA, and no cluster:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_MKL=OFF -DGPU_ENABLED=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

That is what CI runs on every push. Verified on x86-64 Linux and Apple Silicon.

Build options:

| Flag | Default | Effect |
|---|---|---|
| `USE_MKL` | `ON` | Intel MKL reference kernels. Auto-disables with a warning if MKL is absent. |
| `GPU_ENABLED` | `OFF` | CUDA kernels and NVBench GPU benchmarks. Needs the CUDA toolkit. |
| `OPENMP` | `ON` | OpenMP threading for the CPU kernels. |

Full pipeline including GPU, MKL, the Python reference model, and plotting is
in `build_run.sh` (a SLURM batch script, `sbatch build_run.sh`). It expects the
course dataset at `/home/coe4sp4/data` and will not run off the cluster.

---

## Optimizations

### CPU

**GEMM** is cache-blocked over three levels with tile sizes exposed through
`ScheduleParams`, so blocking can be swept from the benchmark harness rather
than recompiled. Tiles of the `(i, j)` iteration space are distributed across
cores with `collapse(2)` to give OpenMP a larger flat iteration space than
row-parallelism alone would. The innermost loop walks contiguous columns of `B`
and `C`, which is the layout the vectorizer wants.

**GEMV** is row-parallel with a vectorized dot-product reduction. It is memory
bound, so the work is in streaming cleanly rather than in blocking.

**SpMM / SpMV** use CSR. SpMM parallelizes across rows of `A` and, for each
stored non-zero, does a vectorized AXPY over a full row of `B`. This keeps the
inner loop dense and contiguous even though the outer structure is sparse,
which is what makes it competitive. Rows are the unit of parallelism, so no
atomics or reductions are needed on the output.

Vectorization is driven by `#pragma omp simd` against `-mavx2 -mfma
-march=native` rather than hand-written intrinsics, which keeps the kernels
readable and portable to non-x86 targets.

### GPU

**Dense GEMM** (`include/kernels.cuh`) is the most heavily tuned kernel:

- 128×128×8 block tiling with an 8×8 register tile per thread, so each thread
  computes 64 outputs and arithmetic intensity stays high enough to hide global
  memory latency
- Shared-memory staging declared as `As[BM][BK + 1]` / `Bs[BK][BN + 1]` — the
  `+1` pad breaks the power-of-two stride that would otherwise serialize a warp
  on shared-memory bank conflicts
- Fully unrolled accumulation over register fragments
- `__restrict__` throughout to let the compiler assume no aliasing
- Coalesced global loads, with load indices computed so consecutive threads hit
  consecutive addresses

**Dense GEMV** assigns one warp per output row. Each lane strides through the
row, and the partial sums are combined with a `__shfl_down_sync` butterfly
reduction, so the reduction never touches shared memory. This is the kernel
that edges out cuBLAS at 256×256.

**Sparse kernels** use CSR, with a different decomposition for each operation
because the two have different reduction shapes:

- **SpMM** maps one thread block to each row of `A`. Threads stride across the
  columns of `C`, so consecutive lanes read consecutive elements of `B` and
  loads stay coalesced. Each thread accumulates in a register and writes its
  output once, which removes any need for atomics.
- **SpMV** maps one warp to each row, with lanes striding over that row's
  non-zeros and a `__shfl_down_sync` reduction producing the result. Rows in a
  pruned weight matrix are short, so a warp is the right granularity where a
  full block would leave most threads idle.

---

## Testing

26 unit tests across 6 GTest suites, covering every kernel and both networks:

| Suite | Coverage |
|---|---|
| `gemm_test` | identity, non-square, accumulation into `C` |
| `gemv_test` | correctness and accumulation |
| `spmv_test` | CSR structure, empty rows, correctness |
| `spmm_test` | CSR × dense, multi-column output |
| `denseNN_test` | end-to-end forward pass, activations, `argmax` |
| `sparseNN_test` | sparse forward pass, agreement with the dense network |

Each suite links only the translation units it exercises, so a failure isolates
to one kernel.

---

## Layout

```
include/         Kernel headers, CUDA kernels (kernels.cuh), shared types
src/             CPU kernels + networks (.cpp), GPU networks (.cu)
test/            GTest suites
script/          Python reference model, magnitude pruning, plotting
main_bench.cpp   CPU kernel benchmarks (Google Benchmark)
main_bench.cu    GPU kernel benchmarks (NVBench)
nn_cpu_bench.cpp CPU network benchmarks
nn_gpu_bench.cu  GPU network benchmarks
build_run.sh     SLURM pipeline for the ECE cluster
```

---

## Honest limitations

- The dense GPU GEMM does not beat cuBLAS. Closing that gap needs
  double-buffered `cp.async` pipelining and tensor cores, neither of which is
  implemented here.
- The sparse network beats dense cuBLAS only at small batch sizes. Past batch
  1000 the irregular memory access from scattered non-zeros outweighs the
  arithmetic saved, and it drops to 0.31–0.57×.
- Pruning is magnitude-based only. A structured or block-sparse scheme would
  suit the GPU memory hierarchy better than unstructured sparsity does.
- Benchmarks come from single-node runs on one machine and were not repeated
  across nodes, so treat small differences as noise.

---

## Attribution

Built for COMPENG 4SP4 (High-Performance Programming), McMaster University.
The course supplied API headers, benchmark scaffolding, and kernel stubs; the
kernel implementations, CUDA code, tests, and tooling are mine. See
[NOTICE.md](NOTICE.md) for the file-level breakdown and licensing.
