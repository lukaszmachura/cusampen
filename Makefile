all: cs

cs: cs.cu	
	nvcc --gpu-architecture=compute_35 cs.cu -o cs

data:
	for ((i=0; i<23242; i++)); do echo 1; done > data.dat
