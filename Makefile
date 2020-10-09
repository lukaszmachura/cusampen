all: cs

cs: cs.cu
	nvcc --gpu-architecture=compute_35 cs.cu -o cs
    
clean:
	rm -rf cs
    
mrproper:
	rm -rf cs *.dat

data:
	awk BEGIN'{srand(${RANDOM}); for (i=0; i<30000; i++) print rand()}' > data.dat
