#include <cusampen/cusampen.h>

using namespace cusampen;

int main(int argc, char **argv){
  params p(argc, argv);
  p.dump();

  int N = countlines(p.infile.c_str());
  if (N <= p.m + 1){
      printf("m (%d) > length of data (%d)\nExiting...", p.m, N);
      return -1;
  }
  
  cuda::init();
  cuda::device gpu(0);
  cuda::primary_context ctx(gpu);
  ctx.set_current(); 

  size_t size = N * sizeof(float);
  cuda::host_ptr<float> x(size);
  cuda::host_ptr<int> mcounts(size);
  cuda::host_ptr<int> mpocounts(size);

  cuda::device_ptr<float> dx(size);
  cuda::device_ptr<int> dmcounts(size);
  cuda::device_ptr<int> dmpocounts(size);

  load_data(p.infile.c_str(), x);
  x.to_device(dx, size);
  ctx.set_current();

  float eps = standard_deviation(x, N) * p.r;
  printf("eps = %f\n", eps);

  const std::string source = get_kernel_source(p.m, eps, N);
  const std::string ptx = compile(source);

  cuda::module mod(ptx);
  cuda::function kernel = mod.get_function("findvec");
  int regs = kernel.get_attribute(CU_FUNC_ATTRIBUTE_NUM_REGS);
  printf("regs %d\n", regs);

  cuda::dim3 threads(128);
  cuda::dim3 blocks((N + 127) / 128);

  void *args[] = {&dx, &dmcounts, &dmpocounts};

  cuda::stream s;
  kernel.launch(blocks, threads, 0, s, args);
  s.synchronize();

  dmcounts.to_host(mcounts, size);
  dmpocounts.to_host(mpocounts, size);
  ctx.synchronize();

  int mcount = reduce(mcounts, N - p.m + 1, p.apen);
  int mpocount = reduce(mpocounts, N - p.m, p.apen);

  FILE * outFile;
  outFile = fopen(p.outfile.c_str(), "w");
  fprintf(outFile, "m vector matches: %lu\n", mcount);
  fprintf(outFile, "(m+1) vector matches: %lu\n", mpocount);
  fprintf(outFile, "ratio = n_{m+1}/n_m: %f\n", (float)mpocount / mcount);
  fprintf(outFile, "SampEn = -ln(ratio): %f\n", -log((float)mpocount / mcount));
  fclose(outFile);
}

