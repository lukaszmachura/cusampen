CC=nvcc

all: cs main

cs: cs.cu
	$(CC) -arch=compute_50 -code=sm_50 -Xptxas -v cs.cu -o cs

main: main.cpp
	g++ -std=c++1y -march=native -fopenmp main.cpp -o main -lm -I. -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -lnvrtc -lcuda

clean:
	rm -rf cs main
    
mrproper:
	rm -rf cs *.dat

data:
	awk BEGIN'{srand(${RANDOM}); for (i=0; i<30000; i++) print rand()}' > data.dat
