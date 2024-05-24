// CUDA kernel to apply ReLU activation element-wise
__global__ void reluKernel(float *d_in, float *d_out, int size) {
    // Global thread index
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Out-of-bounds check
    if (tid < size) {
        d_out[tid] = (d_in[tid] > 0) ? d_in[idx] : 0;
    }
}