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
| **Version 1:** interleaved addressing with warp divergence (branching threads) | 1.64               | 40.9                       | 12.8%                           |
| **Version 2:** interleaved addressing with shared memory bank conflicts | 1.24               | 54.0                          | 16.9%                               |
| **Version 3:** sequential addressing | 0.94              | 71.4                          | 22.3%                               |
| **Version 4:** sequential addressing where first addition happens during load into shared memory | 0.55            | 122                          | 38.1%                               |
| **Version 5:** unrolling the last warp by utilising SIMD (special case of SIMT) | 0.35            | 191.7                          | 59.9%                               |

### Version 1: Interleaved Addressing with Warp Divergence
If threads within a warp execute in lockstep i.e. same instruction at the same time, what happens when there is warp divergence?



### Version 5: Unrolling the Last Warp
> 💡 CUDA guarantees that **threads within the same warp execute in lockstep**, i.e. each thread executes the same instruction at the same time.

Consider the code snippet:
```cpp
__device__ void warpReduce(volatile int* sdata, int ltid) {
        sdata[ltid] += sdata[ltid + 32];
        sdata[ltid] += sdata[ltid + 16];
        sdata[ltid] += sdata[ltid + 8];
        sdata[ltid] += sdata[ltid + 4];
        sdata[ltid] += sdata[ltid + 2];
        sdata[ltid] += sdata[ltid + 1];
}
```
**Question**: for (say) `sdata[0] += sdata[1]`, doesn't the value of `sdata[1]` depend on whether the thread that updates `sdata[1]` has executed?

This question concerns a key aspect of CUDA programming related to how warp-level operations behave. Let us explore this to clarify how synchronization and execution order within a warp ensure data consistency.

- **Warp Execution and Synchronization**:
    - CUDA guarantees that threads within a single warp execute in lockstep, i.e. they execute the same instruction at the same time. 
    - Due to this synchronous execution within warps, certain operations, especially those involving shared memory or registers where each thread reads and writes to its own distinct memory location, do not require explicit synchronization mechanisms like `__syncthreads()`.
- **Reduction Operation**:
    - In `warpReduce`, the reduction process is designed such that each thread adds a value from another thread within the same warp. Here’s how synchronization implicitly happens:
    - The key to the correctness of this approach lies in the reduction steps being carried out in an order that respects data dependencies. Before you can do `sdata[0] += sdata[1]`, the values at `sdata[1]`, `sdata[2]`, and so on, must already be updated in previous steps (like `sdata[1] += sdata[17]`, `sdata[1] += sdata[9]`, `sdata[1] += sdata[5]`, `sdata[1] += sdata[3]`, and `sdata[1] += sdata[2]`). Each step reduces the range of active threads, but all threads perform their updates simultaneously.
    - This lockstep execution means that when a thread performs `sdata[0] += sdata[1]`, the value in `sdata[1]` has already been finalized by its own set of additions, which were part of the same instruction across the warp. Thus, there is no need for synchronization like `__syncthreads()` within these steps because there is no chance that `sdata[1]` is being updated at the same time that it is being read by another thread in the warp.
    - The absence of memory conflicts and the guarantee that each thread reads and writes its own unique memory location before moving on means each step is safe from race conditions within the warp.

### Version 6: Cooperative Groups
- **Thread Group**: the fundamental type in Cooperative Groups is `thread_group`, which is a **handle** to a **group of threads**. 
    - `size()` returns the total number of threads comprising the group.
    - `thread_rank()` returns the index (between `0` and `size()-1`) of the calling thread within a group.
    - `is_valid` checks the validity of a group. 
- **Thread Group Collective Operations**
    - You can synchronize a group by calling its *collective* `sync()` method or using `cooperative_groups::sync()`. 

### References
- [Optimizing Parallel Reduction in CUDA (by Mark Harris)](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [GPU Sum Reduction](https://github.com/mark-poscablo/gpu-sum-reduction/tree/master)
- [Cooperative Groups: Flexible CUDA Thread Programming](https://developer.nvidia.com/blog/cooperative-groups/)
