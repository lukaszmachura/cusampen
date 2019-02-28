#include <math.h>
#include <stdio.h>
#include <getopt.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess){
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}


#define BLOCKSIZE 256

__device__ int d_m;
__device__ float d_r;

// Kernel functions
__global__ 
void reduce(int *g_idata, int *g_out) {
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
  return fabs(a - b) < eps ? true : false;
}

__global__
void findvec(float *base_vec, float *in, int *mvec, int *mplus1vec)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  // check if vec of size m is similar
  int m = true;
  for (int i = 0; i < d_m; i++){
    if (!is_equal(in[index + i], base_vec[i], d_r)){
      m = false;
      break;
    }
  }
  if (m) mvec[index] = 1;

  // check if vec of size m + 1 is similar
  int mplus1 = true;
  for (int i = 0; i <= d_m; i++){ 
    if (!is_equal(in[index + i], base_vec[i], d_r)){
      mplus1 = false;
      break;
    }
  }
  if (mplus1) mplus1vec[index] = 1;
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

    char z;
    int linenumbers = 0;
    while((z = fgetc(f)) != EOF)
	if (z == '\n')
	    linenumbers++;
    fclose(f);
    return linenumbers;
}

// parameters
static struct option options[] = {
    {"in", required_argument, NULL, 'i'},
    {"out", required_argument, NULL, 'o'},
    {"embed", required_argument, NULL, 'm'},
    {"rad", required_argument, NULL, 'r'},
};

void usage(char **argv)
{
    printf("Usage: %s <params> \n\n", argv[0]);
    printf("Model params:\n");
    printf("   -m, --embed=INT        set the embedding dimension 'dim' to INT\n");
    printf("   -r, --radius=FLOAT     set the maximal distance between vectors\n");
    printf("                          to 'radius' to FLOAT\n");
    printf("   -i, --in=FILE_NAME     set the input data to FILE_NAME\n");
    printf("   -o, --out=FILE_NAME    set the output data to FILE_NAME\n")
    printf("\n");
}

def std(float x)
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

int main(void)
{
  int i;
  float *x;
  int *mvec, *mplus1vec;
  int *mmatches, *mplus1matches;
  float *base_vec, r, sd;
  int m;

  // data
  char fname[] = "./data.dat";
  int N = countlines(fname);
  if (N <= 0) return N;

  // parrallelism
  int blockSize = BLOCKSIZE; //4 ;//256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  // Allocate Unified Memory – accessible from CPU or GPU
  gpuErrchk(cudaMallocManaged(&x, N * sizeof(float)));
  gpuErrchk(cudaMallocManaged(&mmatches, sizeof(int)));
  gpuErrchk(cudaMallocManaged(&mplus1matches, sizeof(int)));
  gpuErrchk(cudaMallocManaged(&mvec, N * sizeof(int)));
  gpuErrchk(cudaMallocManaged(&mplus1vec, N * sizeof(int)));

  // initialize data
  load_data(fname, x);
  sd = std(x, N);

  // Sampen algorithm initialisation
  m = 2;
  gpuErrchk(cudaMemcpyToSymbol(d_m, &m, sizeof(int), 0, cudaMemcpyHostToDevice));
  r = 0.2f * sd;
  gpuErrchk(cudaMemcpyToSymbol(d_r, &r, sizeof(float), 0, cudaMemcpyHostToDevice));

  // space in shared mem for base vec (m + 1)
  gpuErrchk(cudaMallocManaged(&base_vec, (m + 1) * sizeof(float)));
  
  // search for EACH possible base vec
  int n_m = 0,
      n_mplus1 = 0;
  for (int ibv = 0; ibv < N - m - 1; ibv++){
      // clean storage
      for (i = 0; i < N; i++){ 
          mvec[i] = 0;
          mplus1vec[i] = 0;
      }
 
      // build temporary base vec of (m + 1) length
      for (i = ibv; i <= m; i++) base_vec[i] = x[i];

      // Run kernel on the GPU
      // find matches for temporary vec
      findvec<<<numBlocks, blockSize>>>(base_vec, x, mvec, mplus1vec);
      gpuErrchk(cudaDeviceSynchronize());

      //reduce
      mmatches[0] = 0;
      reduce<<<numBlocks, blockSize>>>(mvec, mmatches);
      gpuErrchk(cudaDeviceSynchronize());
      n_m += mmatches[0] - 1;

      mplus1matches[0] = 0;
      reduce<<<numBlocks, blockSize>>>(mplus1vec, mplus1matches);
      gpuErrchk(cudaDeviceSynchronize());
      n_mplus1 += mplus1matches[0] - 1;
  }//end of search

  fprintf(stdout, "m vector matches: %d\n", n_m);
  fprintf(stdout, "(m+1) vector matches: %d\n", n_mplus1);
  fprintf(stdout, "ratio = n_{m+1}/n_m: %f\n", (float)n_mplus1/n_m);
  fprintf(stdout, "SampEn = -ln(ratio): %f\n", -log((float)n_mplus1/n_m));

  // Free memory
  gpuErrchk(cudaFree(x));
  gpuErrchk(cudaFree(mmatches));
  gpuErrchk(cudaFree(mplus1matches));
  gpuErrchk(cudaFree(mvec));
  gpuErrchk(cudaFree(mplus1vec));
  
  return 0;
}


//not used kernel functions
//to be removed
__global__
void czek(int n, int *idx, int *str, int *bI, int *bD, int *tI, int *gD)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride){
      idx[i] = index;
      str[i] = stride;
      bI[i] = blockIdx.x;
      bD[i] = blockDim.x;
      tI[i] = threadIdx.x;
      gD[i] = gridDim.x;
  }
}


