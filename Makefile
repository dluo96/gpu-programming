hello_world:
	nvcc src/hello_world.cu -o hello_world -run

saxpy:
	nvcc src/saxpy.cu -o saxpy -run

device_info:
	nvcc src/device_info.cu -o device_info -run

matmul:
	nvcc src/matmul.cu -o matmul -run
