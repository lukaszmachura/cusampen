#ifndef CUSAMPEN_DATA_H
#define CUSAMPEN_DATA_H

#include <stdio.h>
#include <math.h>


namespace cusampen {
static int load_data(const char *fname, float *x)
{
  FILE *f = fopen(fname, "r");
  float buf;
  int i = 0;
  while(fscanf(f, "%f", &buf) > 0)
    x[i++] = buf;
  fclose(f);
  return i - 1;
}

static int countlines(const char *fname)
{
    FILE *f = fopen(fname, "r");
    if (f == NULL)
        return -1;

    char z, buf;
    int linenumbers = 0;
    while((z = fgetc(f)) != EOF)
        if (z == '\n')
            ++linenumbers;
        buf = z;
    
    printf("last: %i\n", buf);
    fclose(f);
    return linenumbers;
}

float standard_deviation(float *x, int N)
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

}
#endif 
