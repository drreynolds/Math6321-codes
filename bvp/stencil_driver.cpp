/* Main routine to run a stencil-based finite-difference method for solution of a
   second-order, scalar-valued BVP:

      u'' = p(t)*u' + q(t)*u + r(t),  a<t<b,
      u(a) = ua,  u(b) = ub

   where the problem has stiffness that may be adjusted using
   the real-valued parameter lambda<0 [read from the command line]


   This driver attempts to solve the problem using a second-order, stencil-based
   finite-difference approximation.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#include <iostream>
#include <iomanip>
#include <vector>
#include <armadillo>
#include "bvp.hpp"

using namespace std;
using namespace arma;

#ifndef USE_SPARSE
#define SPARSE_LINALG 0
#else
#define SPARSE_LINALG 1
#endif

// main routine
int main(int argc, char **argv) {

  // get lambda from the command line, otherwise set to -10
  double lambda;
  if (argc > 1)
    lambda = atof(argv[1]);
  if (lambda >= 0.0)
    lambda = -10.0;

  // define BVP object
  BVP bvp(lambda);

  // loop over spatial resolutions for tests
  vector<int> N = {100, 1000, 10000};
  for (int i=0; i<N.size(); i++) {

    // output problem information
    cout << "\nStencil-based FD method for BVP with lambda = "
         << lambda << ":" << ",  N = " << N[i] << endl;

    // compute/store analytical solution
    vec t = linspace(bvp.a, bvp.b, N[i]+1);
    double h = t(1)-t(0);
    vec utrue(N[i]+1);
    for (int j=0; j<=N[i]; j++)
      utrue(j) = bvp.utrue(t(j));

    // create matrix and right-hand side vectors
#ifdef SPARSE_LINALG
    sp_mat A(N[i]+1,N[i]+1);
#else
    mat A(N[i]+1,N[i]+1);
    A.fill(0.0);
#endif
    vec b(N[i]+1);
    b.fill(0.0);

    // set up linear system
    A(0,0) = 1.0;
    b(0) = bvp.ua;
    A(N[i],N[i]) = 1.0;
    b(N[i]) = bvp.ub;
    for (int j=1; j<N[i]; j++) {
      A(j,j-1) = -1.0 - 0.5*h*bvp.p(t(j));
      A(j,j) = 2.0 + h*h*bvp.q(t(j));
      A(j,j+1) = -1.0 + 0.5*h*bvp.p(t(j));
      b(j) = -h*h*bvp.r(t(j));
    }

    // solve linear system for BVP solution
#ifdef SPARSE_LINALG
    vec u = arma::spsolve(A,b);
#else
    vec u = arma::solve(A,b);
#endif

    // output maximum error
    vec uerr = abs(u-utrue);
    cout << "  Maximum BVP solution error = " << std::setprecision(4)
       << uerr.max() << "\n";
  }

  return 0;
}
