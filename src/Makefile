KERNELS_DIR = kernels

INSTALL_PATH = /usr/local

.PHONY: kernels

all: kernels shared shared-color

install :
	cp libftvp.so $(INSTALL_PATH)/lib
	cp ftvp.cuh $(INSTALL_PATH)/include
	cp memory.cuh $(INSTALL_PATH)/include
	cp libftvp-color.so $(INSTALL_PATH)/lib
	cp ftvp-color.cuh $(INSTALL_PATH)/include
	cp memory-color.cuh $(INSTALL_PATH)/include
	@echo "You may need to run ldconfig as super-user."

uninstall :
	rm -f $(INSTALL_PATH)/lib/libftvp.so
	rm -f $(INSTALL_PATH)/include/ftvp.cuh
	rm -f $(INSTALL_PATH)/include/memory.cuh
	rm -f $(INSTALL_PATH)/lib/libftvp-color.so
	rm -f $(INSTALL_PATH)/include/ftvp-color.cuh
	rm -f $(INSTALL_PATH)/include/memory-color.cuh

kernels :
	$(MAKE) -C $(KERNELS_DIR)

shared : kernels ftvp.cu
	nvcc -Xcompiler -fPIC -dc memory.cu
	nvcc -Xcompiler -fPIC -dc ftvp.cu
	nvcc -o libftvp.so --shared -Xcompiler -fPIC kernels/kernels-2d.o ftvp.o memory.o

shared-color : kernels ftvp-color.cu
	nvcc -Xcompiler -fPIC -dc memory-color.cu
	nvcc -Xcompiler -fPIC -dc ftvp-color.cu
	nvcc -o libftvp-color.so --shared -Xcompiler -fPIC kernels/kernels-2d-color.o ftvp-color.o memory-color.o

clean :
	$(MAKE) -C $(KERNELS_DIR)	clean
	rm -f libftvp.so
	rm -f ftvp.o
	rm -f memory.o
	rm -f libftvp-color.so
	rm -f ftvp-color.o
	rm -f memory-color.o
