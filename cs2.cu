#include <math.h>
#include <stdio.h>
#include <getopt.h>

#define FILE_SIZE 666
#define BLOCKSIZE 256
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess){
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}


__device__ int d_m;
__device__ float d_r;

// Kernel functions
__global__ 
void reduce(int *g_idata, unsigned int *g_out) {
  __shared__ int sdata[BLOCKSIZE];
  // each thread loads one element from global to shared mem
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  sdata[threadIdx.x] = g_idata[i];

  __syncthreads();
  // do reduction in shared mem
  for (int s=1; s < blockDim.x; s *=2)
  {
    int index = 2 * s * threadIdx.x;
    if (index < blockDim.x)
    {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }

  // write result for this block to global mem
  if (threadIdx.x == 0){
      atomicAdd(g_out, sdata[0]);
  }
}

__device__
bool is_equal(float a, float b, float eps)
{
  return fabs(a - b) <= eps ? true : false;
}

__global__
void findvec(float *base_vec, int m, float *in, int *mvec)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i;
  
  // check if vec of size m is similar
  int found = 1;
  for (i = 0; i < m; i++){
    if (!is_equal(in[index + i], base_vec[i], d_r)){
      found = 0;
      break;
    }
  }
  mvec[index] = found;
}

int load_data(char *fname, float *x)
{
  FILE *f = fopen(fname, "r");
  float buf;
  int i = 0;
  while(fscanf(f, "%f", &buf) > 0)
    x[i++] = buf;
  fclose(f);
  return i - 1;
}

int countlines(char *fname)
{
    FILE *f = fopen(fname, "r");
    if (f == NULL)
        return -1;

    char z, buf;
    int linenumbers = 0;
    while((z = fgetc(f)) != EOF)
        if (z == '\n')
            linenumbers++;
        buf = z;
    
    printf("last: %i\n", buf);
    fclose(f);
    return linenumbers;
}

// parameters
typedef struct {
  float r;                  // max distance
  int m;                    // embed dim
  char infile[FILE_SIZE];   //
  char outfile[FILE_SIZE];  //
  bool apen;                 // 1: approximate not sample entropy
} params;

static struct option options[] = {
    {"in", required_argument, NULL, 'i'},
    {"out", required_argument, NULL, 'o'},
    {"embed", required_argument, NULL, 'm'},
    {"radius", required_argument, NULL, 'r'},
    {"apen", required_argument, NULL, 'a'},
};

void dump_params(params * p)
{
  fprintf(stdout,"#\n");
  fprintf(stdout,"#m:%d\n",p->m);
  fprintf(stdout,"#r:%f\n",p->r);
  fprintf(stdout,"#infile:%s\n",p->infile);
  fprintf(stdout,"#outfile:%s\n",p->outfile);
  fprintf(stdout,"#approx:%d\n",p->apen);
  fflush(stdout);
}

void usage(char **argv)
{
    printf("Usage: %s <params> \n\n", argv[0]);
    printf("Model params:\n");
    printf("   -m, --embed=INT        set the embedding dimension 'dim' to INT\n");
    printf("   -r, --radius=FLOAT     set the maximal distance between vectors\n");
    printf("                          to 'radius' to FLOAT\n");
    printf("   -i, --in=FILE_NAME     set the input data to FILE_NAME\n");
    printf("   -o, --out=FILE_NAME    set the output data to FILE_NAME\n");
    printf("   -a, --apen=INT         calculates Approximate Entropy (1) instead of SampEn (def, 0)\n");
    printf("\n");
}

void parse_arguments(int argc, char **argv, params *p)
{
  int c;

  while( (c = getopt_long(argc, argv, "mrioa:", options, NULL)) != EOF) {
    switch (c) {
      case 'm':
        sscanf(optarg, "%d", &(p->m));
        break;
      case 'r':
        sscanf(optarg, "%f", &(p->r));
        break;
      case 'i':
        strcpy(p->infile, optarg);
        break;
      case 'o':
        strcpy(p->outfile, optarg);
        break;
      case 'a':
        int buf;
        sscanf(optarg, "%i", &buf);
        p->apen = buf;
        break;
    }
  }
}

float sandard_deviation(float *x, int N)
{
  int i;
  float sd = 0, mean = 0;
  
  for (i = 0; i < N; ++i){
    mean += x[i];
    sd += x[i] * x[i];
  }
  
  mean /= N;
  sd = sd / N - mean * mean;
  
  return sqrt(sd);
}

int
find_similar_vectors(float *x, int N, int m, int numBlocks, int blockSize, bool apen=false)
{
    int i; //dummy idx
    
    // space in shared mem for base vec m
    float *base_vec;
    gpuErrchk(cudaMallocManaged(&base_vec, m * sizeof(float)));
    
    // space in shared mem for reduction
    unsigned int *matches;
    gpuErrchk(cudaMallocManaged(&matches, sizeof(unsigned int)));
    
    // space in shared mem for identical vectors
    int *mvec;
    gpuErrchk(cudaMallocManaged(&mvec, N * sizeof(int)));

    // search for EACH possible base vec of length m + 1
    unsigned long n = 0;
    for (int ibv = 0; ibv <= N - m; ibv++){
      // build temporary base vec of length m
      for (i = ibv; i < ibv + m; ++i) base_vec[i - ibv] = x[i];

      // matches per node
      for (i = 0; i < N; ++i) mvec[i] = 0;  

      // Run kernel on the GPU
      // find matches for temporary vec
      findvec<<<numBlocks, blockSize>>>(base_vec, m, x, mvec);
      gpuErrchk(cudaDeviceSynchronize());

      //reduce
      matches[0] = 0;
      reduce<<<numBlocks, blockSize>>>(mvec, matches);
      gpuErrchk(cudaDeviceSynchronize());
      n += matches[0] - (apen ? 0 : 1);
    }
    
    gpuErrchk(cudaFree(matches));
    gpuErrchk(cudaFree(mvec));
    
    return n;
}

int main(int argc, char **argv)
{
  params p = {
    0.2f,
    2,
    "data.dat",
    "out.dat",
    false
  };
  
  parse_arguments(argc, argv, &p);
  dump_params(&p);
  
  // data
  int N = countlines(p.infile);
  if (N <= p.m + 1){
      printf("m (%d) > length of data (%d)\nExiting...", p.m, N);
      return N;
  }

  // parrallelism
  int blockSize = BLOCKSIZE; //4 ;//256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  // Allocate Unified Memory – accessible from CPU or GPU
  float *x;
  gpuErrchk(cudaMallocManaged(&x, N * sizeof(float)));

  // initialize data
  load_data(p.infile, x);
  float sd = sandard_deviation(x, N);

  // Sampen algorithm initialisation
  int m = p.m;
  gpuErrchk(cudaMemcpyToSymbol(d_m, &m, sizeof(int), 0, cudaMemcpyHostToDevice));
  float r = p.r * sd;
  gpuErrchk(cudaMemcpyToSymbol(d_r, &r, sizeof(float), 0, cudaMemcpyHostToDevice));

  bool apen = p.apen;
  unsigned long n_m = find_similar_vectors(x, N, m, numBlocks, blockSize, apen);
  unsigned long n_m_plus_1 = find_similar_vectors(x, N, m + 1, numBlocks, blockSize, apen);
       
  FILE * outFile;
  outFile = fopen(p.outfile, "w");
    fprintf(outFile, "m vector matches: %lu\n", n_m);
    fprintf(outFile, "(m+1) vector matches: %lu\n", n_m_plus_1);
    fprintf(outFile, "ratio = n_{m+1}/n_m: %f\n", (float)n_m_plus_1/n_m);
    fprintf(outFile, "SampEn = -ln(ratio): %f\n", -log((float)n_m_plus_1/n_m));
  fclose(outFile);

  // Free memory
  gpuErrchk(cudaFree(x));
  
  return 0;
}
