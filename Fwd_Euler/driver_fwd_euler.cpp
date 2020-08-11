/* Main routine to test the forward Euler method for some scalar-valued ODE problems
     y' = f(t,y), t in [0,5],
     y(0) = y0.

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
    f = -y;
    return 0;
  }
};
//    Convenience function for analytical solution
vec ytrue1(const double t) {
  vec yt(1);
  yt(0) = exp(-t);
  return yt;
};



// Problem 2:  y' = (y+t^2-2)/(t+1)
//    ODE RHS function class -- instantiates a RHSFunction
class RHS2: public RHSFunction {
public:
  int Evaluate(double t, vec& y, vec& f) {    // evaluates the RHS function, f(t,y)
    f(0) = (y(0)+t*t-2.0)/(t+1.0);
    return 0;
  }
};
//    Convenience function for analytical solution
vec ytrue2(const double t) {
  vec yt(1);
  yt(0) = t*t + 2.0*t + 2.0 - 2.0*(t+1.0)*log(t+1.0);
  return yt;
};



// main routine
int main() {

  // time steps to try
  vec h("1.0 0.1 0.01 0.001 0.0001");

  // set problem information
  vec y0_1("1.0");
  vec y0_2("2.0");
  double t0 = 0.0;
  double Tf = 5.0;

  // set desired output times
  int Nout = 6;  // includes initial condition
  vec tspan = linspace(t0, Tf, Nout);

  // create ODE RHS function objects
  RHS1 f1;
  RHS2 f2;

  // create true solution results
  mat Y1true(1,Nout), Y2true(1,Nout);
  for (size_t i=0; i<Nout; i++) {
    Y1true.col(i) = ytrue1(tspan(i));
    Y2true.col(i) = ytrue2(tspan(i));
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

  // problem 2: loop over time step sizes; call stepper and compute errors
  cout << "\nProblem 2:\n";
  ForwardEulerStepper FE2(f2, y0_2);
  for (size_t ih=0; ih<h.n_elem; ih++) {

    // call stepper; output solution and error
    cout << "  h = " << h(ih) << ":\n";
    mat Y = FE2.Evolve(tspan, h(ih), y0_2);

    // output solution, errors, and overall error
    mat Yerr = abs(Y-Y2true);
    for (size_t i=0; i<Nout; i++)
      printf("    y(%.1f) = %9.6f   |error| = %.2e\n",
             tspan(i), Y(0,i), Yerr(0,i));
    cout << "  Max error = " << Yerr.max() << endl;

  }

  return 0;
}
