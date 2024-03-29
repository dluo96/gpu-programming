# cuda-c
Repository for experimenting with CUDA C and C++

This assumes that your host machine has a CUDA-capable GPU and the `nvcc` (included in the Nvidia CUDA Toolkit).

# Usage
Compile with `nvcc`:
```bash
nvcc -o src/saxpy.cu saxpy
```
Run:
```bash
./saxpy
```