# GPUs and CUDA C/C++
This repository is a collection of notes and scripts I am compiling (no pun intended!) in my quest to understand GPUs better. I hope you find some of it useful! 

- [x] Introduces [GPU compute](docs/gpu_compute.md) for CUDA-capable GPUs, covering parallel computing terms (incl. kernels, streaming multiprocessors (SMs), CUDA cores, threads, warps, thread blocks, grids)
- [x] Introduces [GPU memory](docs/GPU_Memory.md) (incl. registers, L1 cache, L2 cache, shared memory, global memory, memory clock rate, memory bus width, peak memory bandwidth).
- [x] Implements a [program](src/device_info.cu) for examining the properties of the attached CUDA GPU(s).
- [x] Implements a ["Hello, World!" program](src/hello_world.cu) with CUDA C/C++.
- [x] Implements a GPU kernel (using CUDA C/C++) for [SAXPY (single-precision A*X Plus Y)](src/saxpy.cu).
- [x] Implements a GPU kernel (using CUDA C/C++) for [matrix multiplication](src/matmul.cu). 
- [x] Implements a GPU kernel (using CUDA C/C++) for [cache tiled matrix multiplication](src/matmul_cache_tiled.cu). 

## Setup
To run the CUDA scripts in this repo, you will need to be set up with a host machine that has a CUDA-enabled GPU and `nvcc` (the NVIDIA CUDA compiler) installed.

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
You can profile with `nvprof`:
```
nvprof ./matmul
```

## References
- [CUDA C++ Programming Guide (v.11.2)](https://docs.nvidia.com/cuda/archive/11.2.0/pdf/CUDA_C_Programming_Guide.pdf)
- [CUDA C Programming Guide (v.9.1)](https://docs.nvidia.com/cuda/archive/9.1/pdf/CUDA_C_Programming_Guide.pdf)
- [Design: GPU vs. CPU](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/design)
- [Performance: GPU vs. CPU](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/performance)
- [Heterogeneous Applications](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/applications)
- [Threads and Cores Redefined](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/threadcore)
- [SIMT and Warps](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/simt_warp)
- [Kernels and SMs](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/kernel_sm)
- [Memory Levels](https://cvw.cac.cornell.edu/gpu-architecture/gpu-memory/memory_levels)
- [Memory Types](https://cvw.cac.cornell.edu/gpu-architecture/gpu-memory/memory_types)
- [An Easy Introduction to CUDA C and C++](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)
- [Introduction to GPU programming with CUDA (C/C++)](https://ulhpc-tutorials.readthedocs.io/en/latest/cuda/)
- [From Scratch: Matrix Multiplication in CUDA](https://www.youtube.com/watch?v=DpEgZe2bbU0)
- [CUDA – Dimensions, Mapping and Indexing](http://thebeardsage.com/cuda-dimensions-mapping-and-indexing/)