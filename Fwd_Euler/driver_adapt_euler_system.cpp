/* Main routine to test adaptive forward Euler method on a system of ODEs
     y' = f(t,y), t in [0,1],
     y(0) = y0.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#include <iostream>
#include "adapt_euler.hpp"

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
  mat eD(size(V),arma::fill::zeros);             // construct the matrix exponential
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

  // solver tolerances
  vec rtols("1.e-3, 1.e-5, 1.e-7");
  vec atol(N);  atol.fill(1.e-13);

  // initial condition
  vec Y0 = randu(N);

  // create true solution object
  mat Ytrue(N,Nout);
  for (size_t i=0; i<Nout; i++)
    Ytrue.col(i) = TrueSolution(V,D,Vinv,tspan(i),t0,Y0);

  // statistics variables
  long int totsteps=0, totfails=0;

  // create adaptive forward Euler solver object (will reset rtol before each solve)
  AdaptEuler AE(MyProblem, 0.0, atol, Y0);

  // loop over tolerances
  cout << "Adaptive Euler test problem, steps and errors vs tolerances:\n";
  for (int ir=0; ir<rtols.size(); ir++) {

    // update the relative tolerance, and call the solver
    cout << "  rtol = " << rtols(ir) << endl;
    AE.rtol = rtols(ir);
    mat Y = AE.Evolve(tspan, Y0);

    // output solution, errors, and overall error
    mat Yerr = abs(Y-Ytrue);

    // output solution, errors, and overall error
    for (size_t i=0; i<Nout; i++)
      cout << "    Y(" << tspan(i) << ") =\t" << trans(Y.col(i));
    cout << "  Overall: "
         << "\t steps = " << AE.steps
	       << "\t fails = " << AE.fails
	       << "\t abserr = " << norm(Yerr,"inf") 
	       << "\t relerr = " << norm(Yerr/Ytrue,"inf") 
	       << endl;

  }

  return 0;
}
