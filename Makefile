CC=/usr/local/cuda-10.0/bin/nvcc

all: cs

cs: cs.cu
	$(CC) --gpu-architecture=compute_35 cs.cu -o cs

clean:
	rm -rf cs
    
mrproper:
	rm -rf cs *.dat

data:
	awk BEGIN'{srand(${RANDOM}); for (i=0; i<30000; i++) print rand()}' > data.dat
