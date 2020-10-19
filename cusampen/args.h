#ifndef CUSAMPEN_ARGS_H
#define CUSAMPEN_ARGS_H

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string>

#define FILE_SIZE 200

namespace cusampen {
static struct option options[] = {
    {"in", required_argument, NULL, 'i'},
    {"out", required_argument, NULL, 'o'},
    {"embed", required_argument, NULL, 'm'},
    {"radius", required_argument, NULL, 'r'},
    {"apen", required_argument, NULL, 'a'},
};

struct params {
  float r;
  int m;
  std::string infile;
  std::string outfile;
  int apen;

  params(int argc, char **argv){
    int c;
    r = 0.2f;
    m = 2;
    infile = "data.dat";
    outfile = "out.dat";
    apen = 0;

    while( (c = getopt_long(argc, argv, "mrioa:", options, NULL)) != EOF) {
      switch (c) {
        case 'm':
          m = atoi(optarg);
          break;
        case 'r':
          r = atof(optarg);
          break;
        case 'i':
          infile = std::string(optarg);
          break;
        case 'o':
          outfile = std::string(optarg);
          break;
        case 'a':
          apen = atoi(optarg);
          break;
      }
    }
  }

  void dump() const
  {
    fprintf(stdout,"#\n");
    fprintf(stdout,"#m:%d\n",m);
    fprintf(stdout,"#r:%f\n",r);
    fprintf(stdout,"#infile:%s\n",infile.c_str());
    fprintf(stdout,"#outfile:%s\n",outfile.c_str());
    fprintf(stdout,"#approx:%d\n",apen);
    fflush(stdout);
  }
};

static void usage(char **argv)
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

}

#endif

