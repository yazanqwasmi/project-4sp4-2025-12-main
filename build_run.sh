#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --export=ALL
#SBATCH --job-name="project"
#SBATCH --nodes=1
#SBATCH --output="project.%j.%N.out"
#SBATCH -t 00:45:00
##################### SLURM (do not change) ^  #####################

# Above are SLURM directives for job scheduling on a cluster,
export SLURM_CONF=/etc/slurm/slurm.conf


echo "----- Building -----"
# Do not change below, it is fixed for everyone
SHAREDDIR=/home/coe4sp4/

find /usr -name "nvcc" 2>/dev/null | head -5

# Source Intel MKL environment
source /opt/intel/oneapi/setvars.sh --force

#cmake -S . -B $(pwd)/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${SHAREDDIR}/libpfm4/ -DPROFILING_ENABLED=ON -DUSE_MKL=ON -DOPENMP=ON
cmake -S . -B $(pwd)/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${SHAREDDIR}/libpfm4/ -DPROFILING_ENABLED=ON -DUSE_MKL=ON -DOPENMP=ON -DGPU_ENABLED=ON
cmake --build $(pwd)/build -- -j8


echo "---- Copying Weights and Dataset -----"

cp -r /home/coe4sp4/data $(pwd)/

echo "---- Running NN, Python----"
# create python virtual environment
python3 -m venv $(pwd)/venv
source $(pwd)/venv/bin/activate
pip install -r $(pwd)/script/requirements.txt
python3 $(pwd)/script/dense_nn.py
python3 $(pwd)/script/sparsify_weight.py


echo "---- Running CPU ----"

mkdir -p $(pwd)/logs
$(pwd)/build/project --benchmark_out="$(pwd)/logs/project.json" --benchmark_out_format=json --benchmark_perf_counters="L1-dcache-loads"
$(pwd)/build/nn_cpu --benchmark_out="$(pwd)/logs/nn_cpu.json" --benchmark_out_format=json


echo "---- Running GPU----"
echo "Note. to run the GPU part, you will need to first enable GPU by add -DGPU_ENABLED=ON"

$(pwd)/build/project_gpu --json "$(pwd)/logs/project_gpu.json" 
$(pwd)/build/nn_gpu --json "$(pwd)/logs/nn_gpu.json" 



echo "---- Running Tests ----"

$(pwd)/build/test/project_gemm_test
$(pwd)/build/test/project_gemv_test
$(pwd)/build/test/project_spmv_test
$(pwd)/build/test/project_spmm_test
$(pwd)/build/test/project_dense_nn_test
$(pwd)/build/test/project_sparse_nn_test

echo "---- Plotting ----"

mkdir -p $(pwd)/plots
python3 $(pwd)/script/plot.py $(pwd)/logs/project.json $(pwd)/logs/nn_cpu.json

