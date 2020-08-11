/* Main routine to test Forward Euler solver for a system of ODEs
     y' = f(t,y), t in [0,5],
     y(0) = y0.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020 */

#include <iostream>
#include "fwd_euler.hpp"

using namespace std;
using namespace arma;


// ODE RHS function class -- instantiates a RHSFunction
//   includes extra routines to set up the problem and
//   evaluate the analytical solution
class ODESystem: public RHSFunction {
  mat A;
public:
  // constructor
  ODESystem(mat& Ain) {
    A = Ain;   // copy input matrix
  }
  // evaluates the RHS function, f(t,y)
  int Evaluate(double t, vec& y, vec& f) {
    f = (A*y);
    return 0;
  }
};

// computes the true solution, y(t)
vec TrueSolution(mat& V, mat& D, mat& Vinv, double t, double t0, vec& y0) {
  mat eD(size(V));             // construct the matrix exponential
  for (size_t i=0; i<D.n_rows; i++)
    eD(i,i) = exp(D(i,i)*(t-t0));
  return (V*(eD*(Vinv*y0)));   // ytrue = V*exp(D*t)*V^{-1}*y0
}



// main routine
int main(int argc, char **argv) {

  // get problem size from command line, otherwise set to 5
  int N = 5;
  if (argc > 1)
    N = atoi(argv[1]);
  cout << "\nRunning system ODE problem with N = " << N << endl;

  // time steps to try
  vec h("0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125");

  // set up problem
  mat V = eye(N,N) + randu(N,N);   // fill with random numbers
  mat D = diagmat(-randu(N,1));
  mat Vinv = inv(V);               // Vinv = V^{-1}
  mat A = (V*(D*Vinv));            // construct system matrix
  cout << "V:\n" << V << endl;
  cout << "Vinv:\n" << Vinv << endl;
  cout << "D:\n" << D << endl;
  cout << "A:\n" << A << endl;
  ODESystem MyProblem(A);
  double t0 = 0.0;
  double Tf = 1.0;

  // set desired output times
  int Nout = 6;  // includes initial condition
  vec tspan = linspace(t0, Tf, Nout);

  // initial condition
  vec Y0 = randu(N);

  // create forward Euler stepper object
  ForwardEulerStepper FE(MyProblem, Y0);

  // loop over time step sizes
  for (size_t ih=0; ih<h.n_elem; ih++) {

    // call stepper
    cout << "\nRunning with stepsize h = " << h(ih) << ":\n";
    mat Y = FE.Evolve(tspan, h(ih), Y0);

    // output solution, errors, and overall error
    double maxabserr = 0.0;
    double maxrelerr = 0.0;
    for (size_t i=0; i<Nout; i++) {
      vec Ytrue = TrueSolution(V,D,Vinv,tspan(i),t0,Y0);
      mat Yerr = abs(Y.col(i)-Ytrue);
      double abserr = norm(Yerr,"inf");
      double relerr = norm(Yerr,"inf") / norm(Ytrue,"inf");
      cout << "  Y(" << tspan(i) << ") =\t" << trans(Y.col(i));
      maxabserr = std::max(maxabserr, abserr);
      maxrelerr = std::max(maxrelerr, relerr);
    }

    cout << "Overall abserr = " << maxabserr
	 << ",  relerr = " << maxrelerr << endl;
  }

  return 0;
}
