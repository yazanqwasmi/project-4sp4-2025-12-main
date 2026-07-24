# 🎓 Project



By the end of the project, you will be able to:
* Optimize and design neural networks.
* Optimize and design sparse neural networks.
* Implement and optimize matrix multiplication.


**Note:** You must review Tutorial 01 before starting this lab. You will need to 
finish tutorial 02 for the GPU part.
***

## 🛠️ Logging in to the ECE Cluster
The ECE computer server is a local server used for this course. The ECE server
has 6 nodes, each with a 20-core Intel CPU and an Ada2000 GPU. The ECE server 
uses a SLURM scheduler. SLURM (Simple Linux Utility for Resource Management)
is an open-source workload manager that allocates exclusive resources (computer nodes)
to users for a specific duration to run their tasks. It manages a queue of jobs, 
ensuring that resources are used efficiently and jobs are run fairly. You will 
need a username to log in to the server.  Your username is the one that you use 
to login to PCs in labs. With that, you will need to SSH to the server using the
following command:

 ```
 ssh <username>@srv-cad.ece.mcmaster.ca
 ```

***

## ➕ Cloning the Repository

This step explains how to clone the repository. You will use **`git`**, a version 
control system, to clone the repository and push your code to it. To clone 
the repository, you will need to use the following command:


```
git clone https://github.com/4sp4-2025/<repo name>.git
```

Where `repo name` is your repository. The repository is private, so you may get an 
error when you enter your username and password. To solve this, you will need to 
 zuse a **Personal Access Token (PAT)** instead of your password.


***

## 🚀 Building and running the starter code

This section explains how you can build and run the code on the 
ECE cluster. First, go to where the project is cloned:

```
cd <where/the/repo/is/cloned>
```
All necessary instructions to build and run the code are 
provided in the `build_run.sh` file.
You can open the file using a text editor like `nano` or `vim`, 
e.g., `vim build_run.sh`.  Use the following command to build 
and run the code:

```
sbatch build_run.sh
```

This command submits the `build_run.sh` script to the SLURM scheduler. 
The instructions in the `build_run.sh` file are
unix commands that will be executed on a compute node. The output of 
the job will be saved in a `*.out`file.

* **Never use `bash`** to run your code on the login node. 
  Use `sbatch` instead. `bash` runs the code directly on
  the login node, while `sbatch` submits the job to the SLURM 
  scheduler, which runs it on an available compute node.
* You should **never** run code on the login node unless it 
  takes less than 30 seconds.
* You can check the status of your job using `squeue -u <username>`.


***


## ✅ Tasks
Deep neural networks (DNN) consist of multiple layers that perform matrix multiplications 
followed by nonlinear operations.
In each layer, the input is multiplied by a weight matrix and then passed through a 
nonlinear activation function (for example, softmax).
The output of a layer becomes the input to the next layer.
Matrix multiplication is the most computationally expensive operation in neural networks.
Your main task is to implement a three-layer neural network.
Implement both dense and sparse variants of the network and compare their accuracy 
and performance against state-of-the-art implementations.



### DNN Description:
A DNN with two linear layers and two non-linear activation functions. 

#### Mathematical Representation:

Let's denote:

`X:` Input vector (n-dimensional)

`W1:` Weights matrix for the first layer (h x n)

`b1:` Biases vector for the first layer (h-dimensional)

`W2:` Weights matrix for the second layer (m x h)

`b2:` Biases vector for the second layer (m-dimensional)

`H:` Hidden layer activations (h-dimensional)

`Z:` Output vector before softmax (m-dimensional)

`Y:` Output vector (m-dimensional)

The forward pass of a DNN with two layers and softmax can be represented as follows:

`H = tanh(X * W1^T + b1)`

`Z = sigmoid(H * W2^T + b2)`

`Y = argmax(Z)`

Where:


tanh is an activation function, defined as:
`softmax(Z) = (exp(Z) - exp(-Z)) / (exp(Z) + exp(-Z))`.
sigmoid is another commonly used activation function in neural networks, defined as `1/(1 + exp(-z))`. argmax returns the index of the maximum element in the vector.

Explanation:

*Input Layer:* The input data is fed into the input layer.

*Hidden Layer:* The input is multiplied by the weights of the first layer, and the biases are added. The result is passed through the tanh activation function.

*Output Layer:* The activations from the hidden layer are multiplied by the weights of the second layer, and the biases are added. The result is passed through the sigmoid activation function to normalize the outputs into a probability distribution.

*Prediction:* Eventually argmax is used to predict the class label based on the output vector Z.


We have already trained the weight matrices and biases for this network on the MNIST dataset.
The weights and biases are included in the `data` folder; they are downloaded from the 
server when you run the provided `build_run.sh` script. See that script for details.
All model parameters are stored as CSV files in the `data/model/` folder.
The MNIST dataset is also provided in the `data` folder as a CSV file.
Functions for reading these CSV files are provided.
The MNIST CSV contains features and labels; labels are located in the first column after loading.

**Note:** The code assumes the data is in the `data` folder. If you set up the project in a 
different location than the server, update the paths accordingly.



### Task 1: Dense Neural Network

- Write a `Python` script (`script/dense_nn.py`) that implements the dense neural network described above. Measure and report the network 
accuracy on the `MNIST` dataset.
- Re-implement the same network in `C++` using your optimized matrix-matrix (`MM`) and matrix-vector (`MV`) operations. 
Measure execution time of the key calls in the dense NN (for example, `MM`, `MV`, and activation functions).
  test only the first 10 features of the `MNIST` dataset.
- Before the end-to-end C++ implementation, develop and optimize the `MM` and `MV` kernels. For `MV` benchmarking.
- Document the optimization strategies applied to each operation and provide an analysis of the resulting performance improvements.


### Task 2: Sparse Neural Network
- Implement a pruning algorithm to sparsify the weight matrices (in `script/sparsify_weight.py`). Magnitude\-based pruning 
(removing weights with the smallest absolute values) is acceptable.

- Write a `Python` script to prune `W1` and `W2` for sparsity levels from 50\% to 95\% in steps of 5\%. 
For each sparsity level, save the pruned weight matrices in dense CSV format using the naming convention: 
`<sparsity_level>_W1.csv` and `<sparsity_level>_W2.csv` (for example, `80_W1.csv` and `80_W2.csv`). 
When loading these files in `C++`, convert the dense matrices to a sparse format (e.g., CSR).

- Implement efficient sparse matrix\-matrix (SpMM) and sparse matrix\-vector (SpMV) kernels. Document the optimization 
strategies for each kernel and provide an analysis of the measured performance improvements.

- After validating SpMV and SpMM, implement the sparse neural network using these operations. Report accuracy and 
runtime for each sparsity level and compare the results with the dense neural network.

### Task 3: GPU Implementation 
Implement both dense and sparse neural networks on the GPU. Use CSR format for sparse matrices on the GPU.
First, implement optimized matrix-matrix (MM) and matrix-vector (MV) kernels on the GPU. Apply common GPU 
optimizations such as memory coalescing, tiling/shared memory, warp-level primitives, proper thread/block configuration, 
asynchronous copies, and tensor cores where applicable.
Then build the dense and sparse neural networks using these kernels and compare their performance on the GPU.

**Important Note** To enable GPU in the project, you will need to use `-DGPU_ENABLED=ON` when calling CMake. 

### Bonus Task: New Pruning Method and Outperforming vendor libraries
* Implement a new pruning method that outperforms magnitude-based pruning in terms of performance while maintaining 
similar accuracy. You may use any method.
* Compare the performance of your dense and sparse neural networks with vendor libraries such as Intel MKL (CPU) and 
cuBLAS/cuSPARSE (GPU). If you outperform these libraries, you will receive bonus points: target 1.2× faster than MKL 
(sparse NN vs dense MKL) and 1.1× faster than cuBLAS (sparse NN vs dense cuBLAS).

### The expected output
Your submission must include the following:

* Implementing dense NN using MM and MV operations (CPU and GPU).
* Implementing sparse NN using SpMM and SpMV operations for different sparsity levels from 50% to 95% with 
a step size of 5% (CPU and GPU).
* Bonus: build a new pruning method that outperforms magnitude-based pruning in terms of performance and provides a similar accuracy.
* Outperforming cuBLAS/cuSPARSE and Intel MKL for dense and sparse NN implementations (bonus).
* Provide stacked bar plots for different optimizations applied to MM/MV and SpMM/SPMV operations. 
  The x-axis should represent different matrix sizes (for MM) and different sparsity levels (for SpMM). 
  The y-axis should represent the execution time. Each bar should be divided into different colors representing 
  different optimizations applied (e.g., naive, cache blocking, vectorization, parallelization, etc.). 
  Provide separate plots for MM/MV and SpMM/SpMV.
* Repeat all above plots for GPU code as well.
* The necessary Python scripts (and packages) to generate all plots. You won't be allowed to push plots to the repo. 
  We should be able to generate all plots using your scripts and the logs generated from running your 
  code on the ECE server.
* four benchmarks are provided in the main directory:
  * `main_bench.cpp`: contains benchmarks for MM/MV/SpMM/SpMV on CPUs using Google Benchmark.
  * `main_bench.cu` : contains benchmarks for MM/MV/SpMM/SpMV on GPUs using NVBench.
  * `nn_cpu_bench.cpp`: contains benchmarks for dense and sparse neural networks on CPUs using Google Benchmark.
  * `nn_gpu_bench.cu`: contains benchmarks for dense and sparse neural networks on GPU.
  
   
### Evaluation
Your lab submission will be graded based on these criteria:

* Your code compiles and runs successfully on the ECE cluster using the provided `build_run.sh` script (Pass/Fail).
* All grading tests are passed as expected.
* The quality of your plots and its analysis, which must:
  * Convey a correct and clear argument.
  * Be visually understandable with proper labels, units, and a concise 1-2 sentence executive summary.
* All plots/model parameters must be generated using your provided Python scripts and the logs from your code runs on the ECE server. 
Plots pushed directly to the repository will not be accepted. 
  

### Submission
Follow these guidelines for a successful submission:
* First, please implement all TODOs in the code and remove all comments that
  start with `TODO`.
* Push all your code to the main branch of your repository before the deadline.
* Ensure your code compiles and runs on the ECE cluster. Submissions that fail to 
 compile or run will receive a grade of zero. The TA will only use the `build_run.sh` file for 
 building and running your code. You may want to make a copy of the script for your final testing.
* The build_run.sh script's execution should only output results from Google Test and 
 Google Benchmark. Do not include any extra output from cout or printf. This can negatively 
 impact your grade, and regrade requests for this reason will not be accepted.
* All logs should be redirected to `logs/` directory, all model parameters should be saved in 
`data/model`, and all plots should be saved in the `plots/` directory. These directories are not 
tracked by git to avoid pushing log files or plots to the repository. 

  
## Descriptive Answers (TODO)
Typically, there is no single correct answer/plot for the following questions. Rely on your thought process!

### Plot(s) 1: CPU MM/MV and SpMV/SpMM performance analysis

![Figure 1: CPU MV/MM performance analysis](plots/Task_1/cpu_kernel_comparison.png)
The CPU kernel results clearly demonstrate how computational complexity and memory behavior differentiate dense and sparse linear-algebra operations. Dense GEMM (matrix–matrix multiplication) is by far the most expensive kernel because it performs **O(M×N×K)** work and generates massive memory traffic, which is why its runtime dominates on a log scale across all tile sizes. In contrast, dense GEMV requires only **O(M×K)** work and streams through memory linearly, producing execution times nearly two orders of magnitude smaller. Our optimized kernels incorporate **OpenMP parallelism** to distribute row-level work across cores, **SIMD vectorization** for inner-loop FMA operations (evident in the GEMM `#pragma omp simd` loops  and in the SpMV reduction loop ), and **cache-aware tiling** which improves locality when reusing blocks of A, B, and C. These optimizations greatly reduce both compute stalls and memory latency, producing the substantial speedups seen in the dense kernels. When shifting to sparse operations (SpMM and SpMV), performance improves further because computation is restricted only to non-zero entries using **CSR format**, avoiding unnecessary multiplications and memory loads. SpMV becomes the fastest kernel overall because each row touches only the non-zero values and performs a simple dot-product, while SpMM remains more expensive due to still having to update full dense rows of the output matrix. Overall, the results show a consistent trend: **GEMM ≫ GEMV > SpMM ≫ SpMV**, and the gains arise directly from reducing arithmetic complexity, improving data locality, exploiting thread-level parallelism, and applying hardware-friendly vectorization.


### Plot(s) 2: sparse / dense NN accuracy/performance analysis
![Figure 2: dense NN accuracy/performance analysis](plots/Task_1/dense_nn_gemm_vs_gemv.png)
![Figure 3: sparse / dense NN accuracy/performance analysis](plots/Task_2/sparse_nn_accuracy_vs_sparsity.png)
The sparse neural network experiments show a clear trade-off between accuracy and performance as weights are pruned from 50% to 95% sparsity. Accuracy remains close to the dense baseline (~92%) up to about 75–80% sparsity, because magnitude-based pruning removes small-magnitude weights that contribute minimally to activations, allowing the network to preserve most of its representational capacity. Beyond 85% sparsity, however, accuracy degrades sharply—eventually falling below 50%—because too many informative connections are removed, effectively reducing the model’s expressive power. In contrast, runtime improves monotonically with sparsity: both SpMM and SpMV become faster because computation is restricted only to the nonzero entries stored in CSR format. SpMM shows the largest absolute speedup but remains slower than SpMV since it processes full batches and still performs dense accumulation, whereas SpMV performs a per-sample sparse mat-vec and benefits most from aggressive pruning. These performance gains stem from several optimizations in the sparse kernels, including **CSR-based compressed storage**, **OpenMP parallelism**, and **SIMD-vectorized activation functions** (e.g., `#pragma omp simd` in tanh/sigmoid) as seen in the sparse NN implementation . Compared to the dense NN implementation, which relies heavily on GEMM/GEMV operations that scale with full matrix dimensions, the sparse NN eliminates unnecessary multiplications and memory loads, causing its runtime to drop by nearly an order of magnitude at high sparsity levels. Overall, the results highlight that sparsity can dramatically accelerate inference while maintaining accuracy—up to a threshold—after which the loss of model capacity outweighs computational gains.


### Plot(s) 3: GPU performance analysis
![Figure 4: GPU GEMM vs GEMV Time](plots/Task_3/gpu_dense_nn_gemm_vs_gemv.png)

This graph compares two dense neural network operations, GEMM (matrix-matrix multiplication) and GEMV (matrix-vector multiplication), across different batch sizes. GEMM performs better at small batch sizes (10-100), but GEMV becomes increasingly slower as the batch size grows, taking over 200ms at a batch size of 60,000 compared to GEMM's approximately 40ms. This performance gap occurs because GEMV processes one vector at a time with less parallelism, while GEMM can process multiple rows and columns simultaneously, making it more efficient for larger workloads.

![Figure 5: GPU GEMM Custom vs CuBLAS](plots/Task_3/gpu_gemm_vs_cublas.png)

This next graph shows a custom matrix multiplication implementation compared to NVIDIA's optimized cuBLAS library, with the left chart displaying execution times and the right showing speedup ratios. The custom implementation is significantly slower across all matrix sizes (256×256, 512×512, 1024×1024), achieving only 29-62% of cuBLAS's speed. cuBLAS maintains its performance advantage because it's heavily optimized, while the custom kernel lacks some advanced optimizations like instruction-level tuning.

![Figure 6: GPU GEMV Custom vs CuBLAS](plots/Task_3/gpu_gemv_vs_cublas.png)

Now this graph shows a custom matrix-vector multiplication performance against cuBLAS across different matrix sizes. At the smallest matrix size (256×256), the custom implementation slightly edges out cuBLAS with a 1.05x speedup, but falls behind at larger sizes (0.88-0.91x). The performance degradation with increasing matrix size suggests the custom kernel's memory access patterns or cache utilization strategies don't scale as well as cuBLAS's optimizations for larger matrices, which likely employ more sophisticated prefetching and memory staging techniques.

![Figure 7: GPU Custom Sparse vs cuSPARSE](plots/Task_3/sparse_nn_vs_cusparse.png)

Here we are comparing custom sparse matrix-matrix multiplication (SpMM) against NVIDIA's cuSPARSE library. Unlike the dense case, the custom SpMM implementation actually beats cuSPARSE across all batch sizes, showing 1.23x to 1.46x speedups. The advantage grows with batch size, achieving the best performance at batch sizes of 10,000 and 60,000. This suggests the custom implementation better handles the irregular memory access patterns inherent in sparse matrices, possibly through more efficient indexing strategies or better thread utilization for scattered data.

Across all experiments, dense operations strongly favour NVIDIA’s optimized libraries, while custom implementations fall short except in the sparse case. The custom SpMM kernel outperforms cuSPARSE because it is tailored to the specific sparsity pattern and workload, giving it an advantage where generic optimizations struggle. Overall, dense workloads benefit from highly tuned vendor libraries, whereas sparse workloads can gain significant speedups from specialized, problem-specific kernels.

![Figure 4: GPU GEMM vs GEMV Time](plots/Task_3/gpu_dense_nn_gemm_vs_gemv.png)

### Plot(s) 4: Bonus (if applicable)
![Figure 8: GPU SPMM vs Dense cuBLAS](plots/Task_3/sparse_nn_vs_dense_cublas.png)

Our final graph comparison shows sparse operations against dense cuBLAS, revealing an important performance trade-off. At small batch sizes (10-100), sparse operations are 2x faster than dense, successfully exploiting sparsity by skipping zero computations. However, at larger batch sizes (1,000+), sparse becomes slower than dense (0.31-0.57x speed), falling below the 1.1x target. This reflects the fundamental trade-off where memory access overhead from scattered non-zero elements eventually dominates the computational savings from skipping zeros, particularly as batch sizes increase and memory access patterns become more irregular.
