/* Main routine to test adaptive forward Euler method on the scalar-valued ODE problem
     y' = -exp(-t)*y, t in [0,5],
     y(0) = 1.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#include <iostream>
#include <iomanip>
#include "adapt_euler.hpp"

using namespace std;
using namespace arma;


// ODE RHS function class -- instantiates a RHSFunction
class MyRHS: public RHSFunction {
public:
  int Evaluate(double t, vec& y, vec& f) {    // evaluates the RHS function, f(t,y)
    f = -exp(-t)*y;
    return 0;
  }
};
//    Convenience function for analytical solution
vec ytrue(const double t) {
  vec yt(1);
  yt(0) = exp(exp(-t)-1);
  return yt;
};


// main routine
int main() {

  // initial condition and time span
  vec y0("1.0");
  double t0 = 0.0;
  double Tf = 5.0;

  // set desired output times
  int Nout = 6;  // includes initial condition
  vec tspan = linspace(t0, Tf, Nout);

  // create ODE RHS function object
  MyRHS rhs;

  // create true solution object
  mat Ytrue(1,Nout);
  for (size_t i=0; i<Nout; i++)
    Ytrue.col(i) = ytrue(tspan(i));

  // tolerances
  vec rtols("1.e-3, 1.e-5, 1.e-7");
  double atol = 1.e-11;

  // create adaptive forward Euler stepper object (will reset rtol before each solve)
  AdaptEuler AE(rhs, 0.0, atol, y0);

  // loop over relative tolerances
  cout << "Adaptive Euler test problem, steps and errors vs tolerances:\n";
  for (size_t ir=0; ir<rtols.size(); ir++) {

    // update the relative tolerance, and call the solver
    cout << "  rtol = " << rtols(ir) << endl;
    AE.rtol = rtols(ir);
    mat Y = AE.Evolve(tspan, y0);

    // output solution, errors, and overall error
    mat Yerr = abs(Y-Ytrue);
    for (size_t i=0; i<Nout; i++)
      printf("    y(%.1f) = %9.6f,   abserr = %.2e,  relerr = %.2e\n",
             tspan(i), Y(0,i), Yerr(0,i), Yerr(0,i)/abs(Y(0,i)));
    cout << "  overall: "
         << "\t steps = " << AE.steps
	       << "\t fails = " << AE.fails
	       << "\t abserr = " << norm(Yerr,"inf") 
	       << "\t relerr = " << norm(Yerr/Ytrue,"inf") 
	       << endl;

  }

  return 0;
}
