# GPU Programming
This repository is a collection of notes, diagrams, and kernels that I am compiling (no pun intended!) to better understand GPU programming. To that end, I focus mainly on implementing GPU kernels in CUDA C and Triton. 

### Notes
- [x] Introduction to [GPU compute](docs/gpu_compute.md) for CUDA-capable GPUs. Covers parallel computing terms including kernels, streaming multiprocessors (SMs), CUDA cores, threads, warps, thread blocks, grids.
- [x] Introduction to [GPU memory](docs/gpu_memory.md). Covers concepts including registers, L1 cache, L2 cache, shared memory, global memory, memory clock rate, memory bus width, peak memory bandwidth.
 

### CUDA C kernels
- [x] ["Hello, World!"](src/hello_world/hello_world.cu).
- [x] [SAXPY (single-precision A*X Plus Y)](src/saxpy/saxpy.cu).
- [x] [Matrix multiplication](src/matrix_multiplication/matmul.cu). 
- [x] [Matrix multiplication with cache tiling](src/matrix_multiplication/matmul_cache_tiled.cu). 
- [x] [Matrix multiplication kernel where each thread computes one row of the output matrix](src/matrix_multiplication/matmul_thread_row.cu).
- [x] [Matrix multiplication kernel where each thread computes one column of the output matrix](src/matrix_multiplication/matmul_thread_col.cu).
- [x] [Matrix-vector multiplication kernel](src/matrix_multiplication/matvec_mul.cu).
- [x] [1D convolution](src/convolution/convolution_1d.cu).
- [x] [1D convolution with constant memory](src/convolution/convolution_1d_constant_memory.cu).
- [ ] 1D convolution with tiling.
- [ ] 2D convolution.
- [x] [Sum reduction: interleaved addressing with warp divergence](src/sum_reduction/sum_reduction_kernels.cu#15).
- [x] [Sum reduction: interleaved addressing with shared memory bank conflicts](src/sum_reduction/sum_reduction_kernels.cu#53).
- [x] [Sum reduction: sequential addressing](src/sum_reduction/sum_reduction_kernels.cu#99).
- [x] [Sum reduction: first sum during load from global memory](src/sum_reduction/sum_reduction_kernels.cu#129).
- [x] [Sum reduction: unrolling of the last warp using SIMD execution](src/sum_reduction/sum_reduction_kernels.cu#177).
- [x] [Sum reduction using Cooperative Groups (CUDA 9 and above)](src/sum_reduction/sum_reduction_cooperative_groups.cu).
- [ ] Pointwise ops: ReLU. 
- [ ] Pointwise ops: ReLU with shared memory.

## Other programs
- [x] [Program](src/device_info.cu) that extracts the properties of the attached CUDA device(s).
- [ ] CUDA Streams. See [here](https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/overlap-data-transfers/async.cu) and [here](https://leimao.github.io/blog/CUDA-Stream/) and [here](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/). 

## Setup
To run the CUDA scripts in this repo, you will need to be set up with a host machine that has a CUDA-enabled GPU and `nvcc` installed.

## Usage
In general, you can compile and execute a CUDA source file as follows:
```bash
nvcc /path/to/source.cu -o /path/to/executable -run
```
For example, you can run the "Hello, World!" kernel using: 
```bash
nvcc src/hello_world.cu -o hello_world -run
```
Note that `.cu` is the required file extension for CUDA-accelerated programs.
See the [Makefile](Makefile) for a more complete list of commands you can run.


### Device query
To query the amount of resources available for your device, run:
```bash
nvcc src/device_info.cu -o device_info -run
```

## References
- [CUDA C++ Programming Guide (v.11.2)](https://docs.nvidia.com/cuda/archive/11.2.0/pdf/CUDA_C_Programming_Guide.pdf)
- [CUDA C Programming Guide (v.9.1)](https://docs.nvidia.com/cuda/archive/9.1/pdf/CUDA_C_Programming_Guide.pdf)
- [Cornell Virtual Workshop: Design: GPU vs. CPU](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/design)
- [Cornell Virtual Workshop: Performance: GPU vs. CPU](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/performance)
- [Cornell Virtual Workshop: Heterogeneous Applications](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/applications)
- [Cornell Virtual Workshop: Threads and Cores Redefined](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/threadcore)
- [Cornell Virtual Workshop: SIMT and Warps](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/simt_warp)
- [Cornell Virtual Workshop: Kernels and SMs](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/kernel_sm)
- [Cornell Virtual Workshop: Memory Levels](https://cvw.cac.cornell.edu/gpu-architecture/gpu-memory/memory_levels)
- [Cornell Virtual Workshop: Memory Types](https://cvw.cac.cornell.edu/gpu-architecture/gpu-memory/memory_types)
- [An Easy Introduction to CUDA C and C++](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)
- [Introduction to GPU programming with CUDA (C/C++)](https://ulhpc-tutorials.readthedocs.io/en/latest/cuda/)
- [CUDA – Dimensions, Mapping and Indexing](http://thebeardsage.com/cuda-dimensions-mapping-and-indexing/)
- [CUDA Crash Course by CoffeeBeforeArch](https://www.youtube.com/playlist?list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU)
- [From Scratch: Matrix Multiplication in CUDA](https://www.youtube.com/watch?v=DpEgZe2bbU0)
- [From Scratch: Cache Tiled Matrix Multiplication in CUDA](https://www.youtube.com/watch?v=ga2ML1uGr5o)
- [Programming Massively Parallel Processors (4th Edition)](https://www.amazon.co.uk/Programming-Massively-Parallel-Processors-Hands/dp/0323912311).