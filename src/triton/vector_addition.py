import triton
import triton.language as tl

@triton.jit
def _add(z_ptr, x_ptr, y_ptr, N):
    # Same as torch.arange
    offsets = tl.arange(0, 1024)

    # Create 1024 pointers to X, Y, Z
    x_ptrs = x_ptr + offsets
    y_ptrs = y_ptr + offsets
    z_ptrs = z_ptr + offsets

    # Load 1024 elements of X, Y, Z
    x = tl.load(x_ptrs)
    y = tl.load(y_ptrs)
    
    # Perform vector addition
    z = x + y

    # Write back 1024 elements of X, Y, and Z
    grid = (1, )
    _add[grid](z, x, y, N)
