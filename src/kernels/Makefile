all : kernels-2d.cu kernels-2d-color.cu
	nvcc -Xcompiler -fPIC -dc kernels-2d.cu
	nvcc -Xcompiler -fPIC -dc kernels-2d-color.cu

clean :
	rm -f kernels-2d.o
	rm -f kernels-2d-color.o
