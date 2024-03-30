# GPU Kernels with CUDA C/C++

## Prerequisites
You have a host machine with
- CUDA-capable GPU(s),
- `nvcc` (Nvidia CUDA compiler), which is included in the Nvidia CUDA Toolkit.


## Usage
Compile with `nvcc`:
```bash
nvcc -o src/saxpy.cu saxpy
```
Run:
```bash
./saxpy
```