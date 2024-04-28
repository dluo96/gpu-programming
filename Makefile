device_info:
	nvcc src/device_analysis/device_info.cu -o device_info -run

hello_world:
	nvcc src/hello_world/hello_world.cu -o hello_world -run

saxpy:
	nvcc src/saxpy/saxpy.cu -o saxpy -run

matmul:
	nvcc src/matrix_multiplication/matmul.cu -o matmul -run

matmul_cache_tiled:
	nvcc src/matrix_multiplication/matmul_cache_tiled.cu -o matmul_cache_tiled -run

matmul_compare:
	nvcc src/matrix_multiplication/matmul.cu -o matmul
	nvcc src/matrix_multiplication/matmul_cache_tiled.cu -o matmul_cache_tiled
	nvprof ./matmul
	nvprof ./matmul_cache_tiled

convolution_1d:
	nvcc src/convolution/convolution_1d.cu -o convolution_1d -run

convolution_1d_constant_memory:
	nvcc src/convolution/convolution_1d_constant_memory.cu \
		-o convolution_1d_constant_memory -run

convolution_1d_compare:
	nvcc src/convolution/convolution_1d.cu -o convolution_1d
	nvcc src/convolution/convolution_1d_constant_memory.cu -o convolution_1d_constant_memory
	nvprof ./convolution_1d
	nvprof ./convolution_1d_constant_memory

sum_reduction:
	nvcc src/sum_reduction/main.cu src/sum_reduction/sum_reduction.cu \
		-o src/sum_reduction/main -run

sum_reduction_v5:
	nvcc src/sum_reduction/sum_reduction_v5.cu -o sum_reduction_v5 -run

clean:
	rm device_info \
		hello_world \
		saxpy matmul \
		matmul_cache_tiled \
		convolution_1d \
		convolution_1d_constant_memory \
		sum_reduction \
		sum_reduction_v5
	