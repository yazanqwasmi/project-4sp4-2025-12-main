# Sparse DNN Inference: CPU and GPU Kernel Engineering

Hand-written dense and sparse linear-algebra kernels for MNIST inference,
benchmarked against Intel MKL, cuBLAS, and cuSPARSE.

The network is a 784 → 512 → 10 MLP (`tanh`, then `sigmoid`, then `argmax`).
Nothing is delegated to a math library: GEMM, GEMV, SpMM, and SpMV are all
implemented from scratch for both CPU and GPU, then measured against the
vendor implementations of the same operations.

A tuned register-tiled GEMM still only reaches 29–62% of cuBLAS throughput.
cuBLAS is hard to beat at dense work, full stop. The custom CSR SpMM tells a
different story: it beats cuSPARSE by 1.23–1.46× across every batch size
tested, because a kernel built for one specific sparsity pattern can skip the
generality cuSPARSE has to pay for.

**Access note:** this was built for COMPENG 4SP4 (High-Performance
Programming) at McMaster University. The full pipeline (MKL, CUDA, and the
benchmarks below) only builds and runs on the McMaster ECE cluster: Intel CPU
nodes plus NVIDIA Ada 2000 GPUs, scheduled through SLURM. You need a McMaster
ECE account and the course dataset to actually execute it, so it isn't
runnable standalone. This README documents the implementation and the
results measured there.

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

**Accuracy vs. sparsity.** Magnitude-based pruning holds accuracy near the
~92% dense baseline up to roughly 75–80% sparsity. Past 85% it falls off
sharply, eventually below 50%, as pruning starts removing genuinely
informative weights. Runtime, meanwhile, improves monotonically with
sparsity. The usable operating point is therefore around 75–80%: most of the
speedup, almost none of the accuracy loss.

**CPU kernel ordering.** Runtime follows `GEMM ≫ GEMV > SpMM ≫ SpMV`, matching
the arithmetic intensity of each: GEMM does O(M·N·K) work while GEMV streams
linearly at O(M·K), and the sparse variants touch only stored non-zeros.

> **Note on figures.** The plots these numbers came from were generated on
> the cluster from benchmark logs and are not reproducible off it (they need
> the course MNIST data, MKL, and CUDA). `script/plot.py` regenerates every
> figure from the benchmark JSON given cluster access.

---

## Building and running

This requires a McMaster ECE cluster account. From a login node:

```bash
ssh <username>@srv-cad.ece.mcmaster.ca
git clone https://github.com/yazanqwasmi/sparse-dnn-hpc.git
cd sparse-dnn-hpc
sbatch build_run.sh
```

`build_run.sh` handles everything: configuring with MKL and CUDA enabled,
building, pulling the MNIST dataset and model weights, running the Python
pruning script, executing the CPU and GPU benchmarks, running the test suite,
and generating plots from the resulting logs. Output lands in `*.out`, logs
in `logs/`, and figures in `plots/`. Both directories are gitignored per the
course's submission rules since they're generated, not committed.

Build options, set via `-D` flags to `cmake` inside `build_run.sh`:

| Flag | Default | Effect |
|---|---|---|
| `USE_MKL` | `ON` | Intel MKL reference kernels for GEMM/GEMV comparisons |
| `GPU_ENABLED` | `ON` | CUDA kernels and NVBench GPU benchmarks |
| `OPENMP` | `ON` | OpenMP threading for the CPU kernels |

---

## Optimizations

### CPU

**GEMM** is cache-blocked in all three dimensions. The two outer tile sizes
come from `ScheduleParams`, so blocking can be swept from the benchmark
harness without recompiling; the `k` blocking factor is fixed at 64. Tiles of
the `(i, j)` iteration space are distributed across cores with `collapse(2)`
to give OpenMP a larger flat iteration space than row-parallelism alone would.
The innermost loop walks contiguous columns of `B` and `C`, which is the
layout the vectorizer wants.

**GEMV** is row-parallel with a vectorized dot-product reduction. It is
memory bound, so the work is in streaming cleanly rather than in blocking.

**SpMM / SpMV** use CSR. SpMM parallelizes across rows of `A` and, for each
stored non-zero, does a vectorized AXPY over a full row of `B`. This keeps the
inner loop dense and contiguous even though the outer structure is sparse,
which is what makes it competitive. Rows are the unit of parallelism, so no
atomics or reductions are needed on the output.

Vectorization is driven by `#pragma omp simd` against `-mavx2 -mfma
-march=native`.

### GPU

**Dense GEMM** (`include/kernels.cuh`) is the most heavily tuned kernel:

- 128×128×8 block tiling with an 8×8 register tile per thread, so each
  thread computes 64 outputs and arithmetic intensity stays high enough to
  hide global memory latency
- Shared-memory staging declared as `As[BM][BK + 1]` / `Bs[BK][BN + 1]`. The
  `+1` pad breaks the power-of-two stride that would otherwise cause bank
  conflicts and serialize a warp's accesses
- Fully unrolled accumulation over register fragments
- `__restrict__` throughout to let the compiler assume no aliasing
- Coalesced global loads, with load indices computed so consecutive threads
  hit consecutive addresses

**Dense GEMV** assigns one warp per output row. Each lane strides through the
row, and the partial sums are combined with a `__shfl_down_sync` butterfly
reduction, so the reduction never touches shared memory. This is the kernel
that edges out cuBLAS at 256×256.

**Sparse kernels** use CSR, with a different decomposition for each operation
because the two have different reduction shapes:

- **SpMM** maps one thread block to each row of `A`. Threads stride across
  the columns of `C`, so consecutive lanes read consecutive elements of `B`
  and loads stay coalesced. Each thread accumulates in a register and writes
  its output once, which removes any need for atomics.
- **SpMV** maps one warp to each row, with lanes striding over that row's
  non-zeros and a `__shfl_down_sync` reduction producing the result. Rows in
  a pruned weight matrix are short, so a warp is the right granularity where
  a full block would leave most threads idle.

---

## Testing

GTest suites cover the CPU kernels and the dense neural network end-to-end:
identity/non-square/accumulation cases for GEMM and GEMV, and a full
forward-pass check (activations through `argmax`) for the dense network.
`build_run.sh` runs the full suite on the cluster as part of the pipeline.

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
The course supplied API headers, benchmark scaffolding, and kernel stubs
(marked in-code as course-provided); the kernel implementations, CUDA code,
tests, and analysis are mine.
