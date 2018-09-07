all: cs

cs: cs.cu	
	nvcc --gpu-architecture=compute_35 cs.cu -o cs

data:
	awk BEGIN'{srand(${RANDOM}); for (i=0; i<30000; i++) print rand()}' > data.dat
