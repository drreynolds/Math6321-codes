/* Main routine to test the forward Euler method for Dahlquist test problem
     y' = lambda*y, t in [0,0.5],
     y(0) = 1,
   for lambda = -100, h in {0.005, 0.01, 0.02, 0.04}

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020 */

#include <iostream>
#include "fwd_euler.hpp"

using namespace std;
using namespace arma;


// Problem 1:  y' = -y
//    ODE RHS function class -- instantiates a RHSFunction
class RHS1: public RHSFunction {
public:
  int Evaluate(double t, vec& y, vec& f) {    // evaluates the RHS function, f(t,y)
    f = -100.0*y;
    return 0;
  }
};
//    Convenience function for analytical solution
vec ytrue1(const double t) {
  vec yt(1);
  yt(0) = exp(-100.0*t);
  return yt;
};


// main routine
int main() {

  // time steps to try
  vec h("0.005 0.01 0.02 0.04 0.08");

  // set problem information
  vec y0_1("1.0");
  double t0 = 0.0;
  double Tf = 0.5;

  // set desired output times
  int Nout = 21;  // includes initial condition
  vec tspan = linspace(t0, Tf, Nout);

  // create ODE RHS function objects
  RHS1 f1;
 
  // create true solution results
  mat Y1true(1,Nout);
  for (size_t i=0; i<Nout; i++) {
    Y1true.col(i) = ytrue1(tspan(i));
  }

  // problem 1: loop over time step sizes; call stepper and compute errors
  cout << "\nProblem 1:\n";
  ForwardEulerStepper FE1(f1, y0_1);
  for (size_t ih=0; ih<h.n_elem; ih++) {

    // call stepper
    cout << "  h = " << h(ih) << ":\n";
    mat Y = FE1.Evolve(tspan, h(ih), y0_1);

    // output solution, errors, and overall error
    mat Yerr = abs(Y-Y1true);
    for (size_t i=0; i<Nout; i++)
      printf("    y(%.1f) = %9.6f   |error| = %.2e\n",
             tspan(i), Y(0,i), Yerr(0,i));
    cout << "  Max error = " << Yerr.max() << endl;

  }

  return 0;
}
