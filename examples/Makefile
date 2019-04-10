all : fast_tv_prox_color

fast_tv_prox_color : fast_tv_prox_color.cpp
	g++ -c fast_tv_prox_color.cpp
	g++ -c -lpng io_png.c	
	g++ -o fast_tv_prox_color fast_tv_prox_color.o io_png.o -L.. -Wl,-rpath='../src/' -lftvp-color -L/usr/local/cuda/lib -lcuda -lpng

clean :
	rm -f fast_tv_prox_color.o io_png.o
	rm -f fast_tv_prox_color
