/* Main routine to run an implicit 3-node Lobatto finite-difference method
   for solution of a second-order, scalar-valued BVP:

      u'' = p(t)*u' + q(t)*u + r(t),  a<t<b,
      u(a) = ua,  u(b) = ub

   where the problem has stiffness that may be adjusted using
   the real-valued parameter lambda<0 [read from the command line]

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

// utility routine to map from physical/component space to linear algebra index space
//    interval:  physical interval index [1 <= interval <= N]
//    location:  location in interval [0=left, 1=midpoint, 2=right]
//    component: solution component at this location [0=u, 1=u']
int idx(int interval, int location, int component) {
  return ( 4*(interval-1) + 2*location + component );
}


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

  // test 'idx' function by outputting mapping for small N
  cout << "Test output from 'idx' function for N = 5, M = 22:";
  for (int interval=1; interval<=5; interval++) {
    cout << "\n  interval " << interval << ", (loc,comp,idx):";
    for (int location=0; location<=2; location++) {
      for (int component=0; component<=1; component++) {
        cout << "  (" << location << ", " << component << ", "
             << idx(interval,location,component) << ")";
      }
    }
  }

  // loop over spatial resolutions for tests
  vector<int> N = {100, 1000, 10000};
  for (int i=0; i<N.size(); i++) {

    // output problem information
    cout << "\nImplicit Lobatto-3 FD method for BVP with lambda = "
         << lambda << ":" << ",  N = " << N[i] << endl;

    // compute/store analytical solution
    vec t = zeros(N[i]+1);
    t(0) = 0.0;
    t(N[i]) = 1.0;
    for (int j=1; j<=N[i]-1; j++)
      t(j) = 0.5*(1.0-cos((2*j-1)*M_PI/(2*(N[i]-1))));
    vec utrue(N[i]+1);
    for (int j=0; j<=N[i]; j++)
      utrue(j) = bvp.utrue(t(j));

    // set integer for overall linear algebra problem size
    int M = 4*N[i]+2;

    // create matrix and right-hand side vectors
#ifdef USE_SPARSE
    sp_mat A(M,M);
#else
    mat A(M,M);
    A.fill(0.0);
#endif
    vec b(M);
    b.fill(0.0);

    // set up linear system:
    //    recall 'idx' usage: idx(interval,location,component)
    //      interval:  physical interval index [1 <= interval <= N]
    //      location:  location in interval [0=left, 1=midpoint, 2=right]
    //      component: solution component at this location [0=u, 1=u']
    A(0,idx(1,0,0)) = 1.0;
    b(0) = bvp.ua;
    A(1,idx(N[i],2,0)) = 1.0;
    b(1) = bvp.ub;
    int irow = 2;
    for (int j=1; j<=N[i]; j++) {

      // setup interval-specific information
      double tl = t(j-1);
      double tr = t(j);
      double th = 0.5*(tl+tr);
      double h = tr-tl;

      // setup first equation for this interval:
      //    -24*y_{j-1,0} - 5*h*y_{j-1,1} + 24*y_{j-1/2,0} - 8*h*y_{j-1/2,1} + h*y_{j,1} = 0
      A(irow,idx(j,0,0)) = -24.0;
      A(irow,idx(j,0,1)) = -5.0*h;
      A(irow,idx(j,1,0)) = 24.0;
      A(irow,idx(j,1,1)) = -8.0*h;
      A(irow,idx(j,2,1)) = h;
      b(irow) = 0.0;
      irow++;

      // setup second equation for this interval:
      //    -5*h*q_{j-1}*y_{j-1,0} - (24+5*h*p_{j-1})*y_{j-1,1} - 8*h*q_{j-1/2}*y_{j-1/2,0}
      //      + (24-8*h*p_{j-1/2})*y_{j-1/2,1} + h*q_{j}*y_{j,0} + h*p_{j}*y_{j,1} = h*(5*r_{j-1}+8*r_{j-1/2}-r_{j})
      A(irow,idx(j,0,0)) = -5.0*h*bvp.q(tl);
      A(irow,idx(j,0,1)) = -(24.0 + 5.0*h*bvp.p(tl));
      A(irow,idx(j,1,0)) = -8.0*h*bvp.q(th);
      A(irow,idx(j,1,1)) = (24.0-8.0*h*bvp.p(th));
      A(irow,idx(j,2,0)) = h*bvp.q(tr);
      A(irow,idx(j,2,1)) = h*bvp.p(tr);
      b(irow) = h*(5.0*bvp.r(tl) + 8.0*bvp.r(th) - bvp.r(tr));
      irow++;

      // setup third equation for this interval:
      //    -6*y_{j-1,0} - h*y_{j-1,1} - 4*h*y_{j-1/2,1} + 6*y_{j,0} - h*y_{j,1} = 0
      A(irow,idx(j,0,0)) = -6.0;
      A(irow,idx(j,0,1)) = -h;
      A(irow,idx(j,1,1)) = -4.0*h;
      A(irow,idx(j,2,0)) = 6.0;
      A(irow,idx(j,2,1)) = -h;
      b(irow) = 0.0;
      irow++;

      // setup fourth equation for this interval:
      //    -h*q_{j-1}*y_{j-1,0} - (6+h*p_{j-1})*y_{j-1,1} - 4*h*q_{j-1/2}*y_{j-1/2,0} - 4*h*p_{j-1/2}*y_{j-1/2,1}
      //       - h*q_j*y_{j,0} + (6-h*p_j)*y_{j,1} = h*(r_{j-1} + 4*r_{j-1/2} + r_{j})
      A(irow,idx(j,0,0)) = -h*bvp.q(tl);
      A(irow,idx(j,0,1)) = -(6.0 + h*bvp.p(tl));
      A(irow,idx(j,1,0)) = -4.0*h*bvp.q(th);
      A(irow,idx(j,1,1)) = -4.0*h*bvp.p(th);
      A(irow,idx(j,2,0)) = -h*bvp.q(tr);
      A(irow,idx(j,2,1)) = (6.0-h*bvp.p(tr));
      b(irow) = h*(bvp.r(tl) + 4.0*bvp.r(th) + bvp.r(tr));
      irow++;

    }

    // solve linear system for BVP solution
#ifdef USE_SPARSE
    vec y = arma::spsolve(A,b);
#else
    vec y = arma::solve(A,b);
#endif

    // output maximum error
    vec u(N[i]+1);
    for (int j=0; j<=N[i]; j++)
      u(j) = y(idx(j+1,0,0));
    vec uerr = abs(u-utrue);
    cout << "  Maximum BVP solution error = " << std::setprecision(4)
       << uerr.max() << "\n";
  }

  return 0;
}
