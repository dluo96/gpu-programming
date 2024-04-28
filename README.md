# GPUs and CUDA C/C++
This repository is a collection of notes and scripts I am compiling (no pun intended!) to better understand GPUs and CUDA programming. Enjoy! 

### Notes
- [x] Introduction to [GPU compute](docs/gpu_compute.md) for CUDA-capable GPUs. Covers parallel computing terms including kernels, streaming multiprocessors (SMs), CUDA cores, threads, warps, thread blocks, grids.
- [x] Introduction to [GPU memory](docs/gpu_memory.md). Covers concepts including registers, L1 cache, L2 cache, shared memory, global memory, memory clock rate, memory bus width, peak memory bandwidth.
- [ ] Introduction to the **NVIDIA CUDA Compiler (NVCC) Driver**.
- [ ] Introduction to the **NVIDIA Nsight Compute CLI (ncu)**.
 

### CUDA C/C++ Kernels
- [x] CUDA C/C++ ["Hello, World!"](src/hello_world/hello_world.cu).
- [x] [SAXPY (single-precision A*X Plus Y)](src/saxpy/saxpy.cu).
- [x] [Matrix multiplication](src/matrix_multiplication/matmul.cu). 
- [x] [Matrix multiplication with cache tiling](src/matrix_multiplication/matmul_cache_tiled.cu). 
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
- [ ] CUDA Streams. See [here](https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/overlap-data-transfers/async.cu) and [here](https://leimao.github.io/blog/CUDA-Stream/) and [here](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/). 

### Analysis Programs
- [x] [Program](src/device_info.cu) that extracts the properties of the attached CUDA device(s).
- [ ] Profiling with `clock()`.

## Setup
To run the CUDA scripts in this repo, you will need to be set up with a host machine that has a CUDA-enabled GPU and `nvcc` installed.

## Usage
### CUDA C/C++ "Hello, World!"
Compile and execute with `nvcc`:
```bash
nvcc src/hello_world.cu -o hello_world -run
```
Note that `.cu` is the required file extension for CUDA-accelerated programs.

### Device Query
```bash
nvcc src/device_info.cu -o device_info -run
```

### SAXPY Kernel with CUDA C/C++
```bash
nvcc src/saxpy.cu -o saxpy -run
```

### Matrix Multiplication Kernel with CUDA C/C++
```
nvcc src/matmul.cu -o matmul -run
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
- [From Scratch: Matrix Multiplication in CUDA](https://www.youtube.com/watch?v=DpEgZe2bbU0)
- [From Scratch: Cache Tiled Matrix Multiplication in CUDA](https://www.youtube.com/watch?v=ga2ML1uGr5o)
- [CUDA – Dimensions, Mapping and Indexing](http://thebeardsage.com/cuda-dimensions-mapping-and-indexing/)