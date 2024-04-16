# Sum Reduction
- **Sum reduction** is a special case of **parallel reduction**, which is a common and important data parallel primitive. 
- Sum reduction adds up all the elements of an array. 
    - While it is easy to implement naively in CUDA C/C++, it is harder to implement optimally.
    - It serves as a great optimization example: we will implement 7 versions, demonstrating 
    important optimization strategies along the way.
- Can parallelise the reduction by letting each thread block reduce a portion of the array. 
    - A tree-based approach can be used within each thread block. 
    - Each thread block would compute a partial result. 
- This raises the question: **how do we communicate partial results between blocks?**

### Global Synchronization
> 💡 CUDA does **not** have global synchronization, so we instead use **kernel decomposition**.

- If we could synchronize across all thread blocks, we could reduce large arrays. 
    - Could do a global sync after each block produces its result.
    - Once all blocks reach sync, continue recursively. 
- BUT, CUDA has no global sync. Why?
    - Expensive to build in hardware for GPUs with high processor count. 
- The solution is to **decompose into multiple kernels**. 
    - The kernel launch serves as a global sync point. 
    - A kernel launch has negligible hardware overhead and low software overhead. 
- In the case of reduction, the code for all levels (kernel invocations) is the same. 
    - Thus, we can use **recursive** kernel invocation. 

### Optimization Goal
> 💡 Our goal is to maximize **bandwidth**. 

- We want to achieve peak GPU performance. 
- What do we mean by "performance"? We need to choose the correct performance metric to maximize:
    - **GFLOPS** is the correct metric to maximize for *compute-bound* kernels.
    - **Bandwidth** is the correct metric to maximize for *memory-bound* kernels.
- Given that reduction operations have low arithmetic intensity (1 flop per element loaded), we should aim to maximize **bandwidth**. 
- We will use a Tesla T4 GPU for this example. 
    - Memory bus width of 256 bits.
    - Memory clock rate of 5.001 GHz with GDDR memory.
    It follows that the *theoretical* **peak bandwidth** is **320.064 GB/s**. We will try to get as close to this as possible. 

## Versions 1-7
### Bandwidth Comparison
- We compared the bandwidth of the 7 sum reduction implementations. 
- All of the experiments were run with the following assumptions:
    - Tesla T4 GPU,
    - Thread block size of 128,
    - `N=2^24` elements.
- Note that:
    - "Time elapsed" refers to the duration of the kernels. 
    - Bandwidth is calculated according to: `Bandwidth (in GB/s) = Total Data Size (in GB) / Time (s)`. 
- The table below summarises the experiments, reporting each number to 3 significant figures. 

| Kernel      | Time Elapsed (ms) | Bandwidth (GB/s) | % of Theoretical Peak Bandwidth (320.064 GB/s) |
|------------------|-----------------|-----------------------------|---------------------------------|
| **Version 1:** interleaved addressing with divergent branching | 1.64               | 40.9                       | 12.8%                           |
| **Version 2:** interleaved addressing with bank conflicts | 1.24               | 54.0                          | 16.9%                               |


### Sum Reduction v1: Interleaved Addressing
- See full implementation [here](../src/parallel_reduction_diverged.cu)



### References
- [Optimizing Parallel Reduction in CUDA (by Mark Harris)](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [GPU Sum Reduction](https://github.com/mark-poscablo/gpu-sum-reduction/tree/master)
