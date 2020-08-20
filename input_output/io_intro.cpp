/* D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

// Inclusions
#include <stdlib.h>
#include <iostream>
#include <armadillo>
using namespace arma;

// Example routine to show how to perform C++ output from Armadillo's
// `mat' and 'vec' classes, followed by input/plotting in Matlab/Python
int main(int argc, char **argv) {

  // get problem size from command line, otherwise set to 201
  int N = 201;
  if (argc > 1)
    N = atoi(argv[1]);
  std::cout << "\nRunning I/O test using vectors of size N = " << N << std::endl;
  
  // create x data
  vec x = linspace(-1.0, 1.0, N);

  // create function data (first 5 odd-degree Chebyshev polynomials)
  mat T(N,5);
  for (int j=0; j<5; j++)
    for (int i=0; i<N; i++)
      T(i,j) = cos((j*2+1.0) * acos(x(i)));

  // save data to disk
  x.save("x.txt", arma::raw_ascii);
  T.save("T.txt", arma::raw_ascii);
  
  std::cout << "Completed writing data to disk: x.txt and T.txt\n\n";
  return 0;
} // end main
