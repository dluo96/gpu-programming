# SAXPY using CUDA C/C++
- SAXPY stands for **single-precision A*X Plus Y**. 
- The complete CUDA C/C++ implementation of SAXPY can be found in [saxpy.cu](src/saxpy.cu).
- The function `saxpy` is the kernel that runs in parallel on the GPU, whereas `main` (the usual C/C++ entry point) is the host code.
    ```cpp
    __global__
    void saxpy(int n, float a, float *x, float *y)
    {
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        if (i < n) y[i] = a*x[i] + y[i];
    }
    ```
- `x` and `y` are pointers to the host arrays, allocated with the familiar `malloc`. 
- `d_x` and `d_y` are pointers to the device arrays allocated with the `cudaMalloc` function from the CUDA runtime API. 
- The host and the device have separate memory spaces, *both of which can be accessed from the host code*. 

- The `saxpy` kernel is launched by the statement
    ```cpp
    saxpy<<<(N+255)/256, 256>>>(N, 2.0, d_x, d_y);
    ```
    where `<<<(N+255)/256, 256>>>` is the execution configuration, which specifies how many GPU threads (and thread blocks) execute the kernel in parallel. The first argument is the number of thread blocks and the second argument is the number of threads per thread block.
- In CUDA, there is a hierarchy of threads *in software* which mimics how CUDA cores are grouped on the GPU hardware. A **grid** consists of **thread blocks**, each of which consists of **threads**. 
- Cleanup: we free allocated device memory with `cudaFree()` and host memory with the familiar `free()`.