/* Main routine to test the backward Euler, trapezoidal, and
   forward Euler methods on the scalar-valued ODE problem
     y' = lambda*y + (1-lambda)*cos(t) - (1+lambda)*sin(t), t in [0,5],
     y(0) = 1.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#include <iostream>
#include "bwd_euler.hpp"
#include "fwd_euler.hpp"

using namespace std;
using namespace arma;


// Define classes to compute the ODE RHS function and its Jacobian

//    ODE RHS function class -- instantiates a RHSFunction
class MyRHS: public RHSFunction {
public:
  double lambda;                              // stores some local data
  int Evaluate(double t, vec& y, vec& f) {    // evaluates the RHS function, f(t,y)
    f(0) = lambda*y(0) + (1.0-lambda)*cos(t) - (1.0+lambda)*sin(t);
    return 0;
  }
};

//    ODE RHS Jacobian function class -- instantiates a RHSJacobian
class MyJac: public RHSJacobian {
public:
  double lambda;                              // stores some local data
  int Evaluate(double t, vec& y, mat& J) {    // evaluates the RHS Jacobian, J(t,y)
    J(0,0) = lambda;
    return 0;
  }
};


// Convenience function for analytical solution
vec ytrue(const double t) {
  vec yt(1);
  yt(0) = sin(t) + cos(t);
  return yt;
};



// main routine
int main() {

  // time steps to try
  vec h("1.0, 0.1, 0.01, 0.001");

  // lambda values to try
  vec lambdas("-1.0, -10.0, -50.0");

  // set problem information
  vec y0("1.0");
  double t0 = 0.0;
  double Tf = 5.0;

  // set desired output times
  int Nout = 6;  // includes initial condition
  vec tspan = linspace(t0, Tf, Nout);

  // create ODE RHS and Jacobian objects
  MyRHS rhs;
  MyJac Jac;

  // create true solution results
  mat Ytrue(1,Nout);
  for (size_t i=0; i<Nout; i++) {
    Ytrue.col(i) = ytrue(tspan(i));
  }

  // create time stepper objects
  BackwardEulerStepper BE(rhs, Jac, y0);
  ForwardEulerStepper  FE(rhs, y0);

  // update Newton solver parameters
  BE.newt.tol = 1.e-3;
  BE.newt.maxit = 20;
  BE.newt.show_iterates = false;

  //------ Backwards Euler tests ------

  // loop over lambda values
  for (int il=0; il<lambdas.n_elem; il++) {

    // set current lambda value into rhs and Jac objects
    rhs.lambda = lambdas(il);
    Jac.lambda = lambdas(il);

    // loop over time step sizes
    for (int ih=0; ih<h.n_elem; ih++) {

      // call stepper
      cout << "\nRunning backward Euler with stepsize h = " << h(ih)
	   << ", lambda = " << lambdas(il) << ":\n";
      mat Y = BE.Evolve(tspan, h(ih), y0);

      // output solution, errors, and overall error
      mat Yerr = abs(Y-Ytrue);
      for (size_t i=0; i<Nout; i++)
        printf("    y(%.1f) = %9.6f   |error| = %.2e\n",
               tspan(i), Y(0,i), Yerr(0,i));
      cout << "  Max error = " << Yerr.max() << endl;

    }
    cout << endl;
  }


  //------ Forward Euler tests ------

  // loop over lambda values
  for (int il=0; il<lambdas.n_elem; il++) {

    // set current lambda value into rhs and Jac objects
    rhs.lambda = lambdas(il);
    Jac.lambda = lambdas(il);

    // loop over time step sizes
    for (int ih=0; ih<h.n_elem; ih++) {

      // call stepper
      cout << "\nRunning forward Euler with stepsize h = " << h(ih)
	   << ", lambda = " << lambdas(il) << ":\n";
      mat Y = FE.Evolve(tspan, h(ih), y0);

      // output solution, errors, and overall error
      mat Yerr = abs(Y-Ytrue);
      for (size_t i=0; i<Nout; i++)
        printf("    y(%.1f) = %9.6f   |error| = %.2e\n",
               tspan(i), Y(0,i), Yerr(0,i));
      cout << "  Max error = " << Yerr.max() << endl;

    }
    cout << endl;
  }

  return 0;
}
