"""Adapted from the Triton documentation (https://triton-lang.org/main/index.html)."""
import torch

import triton
import triton.language as tl
from triton.runtime import driver


device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]     # Number of streaming multiprocessors (SMs)
NUM_REGISTERS = properties["max_num_regs"]      # Maximum number of registers
SIZE_SMEM = properties["max_shared_mem"]        # Size of shared memory in bytes
WARP_SIZE = properties["warpSize"]              # Number of threads in a warp
target = triton.runtime.driver.active.get_current_target()
kernels = {}

# import os
# os.environ["TRITON_INTERPRET"] = "1"

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows, 
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr
):
    # Assuming the softmax calculation is done per row, the rows are independent.
    # Thus, we can parallelize across rows.

    # Each program starts from a different row. It handles one row at a time but
    # possibly multiple rows overall (depending of the number of programs).
    row_start = tl.program_id(axis=0)
    row_step = tl.num_programs(axis=0)

    # For each program, loop over the rows that the program will handle.
    # Each iteration of the loop corresponds to a program computing the softmax
    # for one row.
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The row stride denotes how much we need to increment the pointer to 
        # advance 1 row. This is usually the number of columns of the input. 
        row_start_ptr = input_ptr + row_idx * input_row_stride

        # The block size was set to be the next power of two greater than `n_cols`, 
        # so we can fit each row in a single block.
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets  # List of pointers to the row elements

        # Load the row into SRAM, using a mask since `BLOCK_SIZE` may be greater than
        # `n_cols`. Note out-of-bound elements won’t affect the sum since exp(-inf)=0.
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)

        # Note that exponentiation in Triton is fast but approximate.
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)  # Sum across the row
        softmax_output = numerator / denominator

        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than 
    # the number of columns in `x`. Thus, each row fits in a single block. 
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Ask the compiler to use more threads per row by increasing the number of warps 
    # (`num_warps`) over which the softmax calculation for each row is distributed. 
    # In this case, each kernel instance (program) will be automatically parallelized
    # to cooperatively execute using 8 * 32 = 256 threads. 
    num_warps = 8

    # Number of stages that the compiler should use when software-pipelining loops
    num_stages = 4 if SIZE_SMEM > 200_000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # Compute the input and output row strides (the jump necessary to go from one 
    # element to the next in the row dimension). For example, we set the input
    # row stride equal to the number of columns of x since this is the number of
    # elements per row. Similarly for the output row stride.
    input_row_stride=x.stride(dim=0)    
    output_row_stride=y.stride(dim=0)

    # Pre-compile kernel to get register usage and compute thread occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(
            x,
            y,
            input_row_stride,
            output_row_stride,
            n_rows,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=num_stages,
            num_warps=num_warps, 
            grid=(1, )
        )

        kernel._init_handles()
        n_registers = kernel.n_regs
        size_smem = kernel.metadata.shared
        occupancy = NUM_REGISTERS // (n_registers * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    # If the number of programs allowed by the the SM count and the occupancy
    # is larger than the number of rows, simply make each program/instance
    # perform softmax for only a single row. In this case, the grid dimension
    # (number of programs/instances) is simply equal to the number of rows.
    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](
        x, y, input_row_stride, output_row_stride, n_rows, n_cols,
    )

    return y

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device='cuda')
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
    print("Success")

    @triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=["Triton", "Torch"],
        styles=[('blue', '-'), ('red', '-')],
        ylabel="Global Memory Bandwidth (GB/s)",
        plot_name="softmax-performance",
        args={'M': 4096},  # Number of columns in input matrix
    ))

    def benchmark(M, N, provider):
        # Initialise input
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)

        # Create and set CUDA stream
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)

        # Benchmark the runtime, getting the mean (default)
        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
        if provider == 'triton':
            ms = triton.testing.do_bench(lambda: softmax(x))
        
        # Compute the global memory bandwidth in GB/s.
        # The factor of 2 is because we have two tensors (input `x` and output `y`). 
        # `x.nelement()` gives the total number of elements in `x`. 
        # `x.element_size()` gives the size in bytes of an individual element of `x`
        # Thus, the formula basically says:
        # Global memory bandwidth in GB/s = (#Bytes)/(Runtime in ms)*(1e-9 GB/B)*(1e3 ms/s)
        gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)

        return gbps(ms)

    benchmark.run(print_data=True, save_path="/home/danielluo/cuda-c/benchmarks/softmax/")
