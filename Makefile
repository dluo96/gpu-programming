device_info:
	nvcc src/device_info.cu -o device_info -run

hello_world:
	nvcc src/hello_world.cu -o hello_world -run

saxpy:
	nvcc src/saxpy.cu -o saxpy -run

matmul:
	nvcc src/matmul.cu -o matmul -run

matmul_cache_tiled:
	nvcc src/matmul_cache_tiled.cu -o matmul_cache_tiled -run

matmul_compare:
	nvcc src/matmul.cu -o matmul
	nvcc src/matmul_cache_tiled.cu -o matmul_cache_tiled
	nvprof ./matmul
	nvprof ./matmul_cache_tiled

convolution_1d:
	nvcc src/convolution_1d.cu -o convolution_1d -run

clean:
	rm device_info hello_world saxpy matmul matmul_cache_tiled
	