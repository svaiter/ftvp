all : pyftvp.so

pyftvp.so :
	python setup.py build_ext --inplace

install :
	python setup.py build_ext
	python setup.py install

clean :
	rm -rf build
	rm -f pyftvp.so pyftvp.c pyftvp.cpp
