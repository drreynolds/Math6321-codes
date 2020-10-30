/* Main routine to test the generic LMM solver method on the
   scalar-valued ODE problem
     y' = lambda*y + (1-lambda)*cos(t) - (1+lambda)*sin(t), t in [0,5],
     y(0) = 1.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#include <iostream>
#include "lmm.hpp"

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
  vec h("0.1, 0.05, 0.01, 0.005, 0.001");

  // storage for errors
  vec e(h.n_elem);

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

  // Adams-Bashforth-2
  cout << "\nAB-2 -- should be O(h^2):\n";
  vec AB2_a("1.0, -1.0, 0.0");
  vec AB2_b("0.0, 1.5, -0.5");
  LMMStepper AB2(rhs, Jac, y0, AB2_a, AB2_b);

  // loop over lambda values
  for (int il=0; il<lambdas.n_elem; il++) {

    // set current lambda value into rhs and Jac objects
    cout << "  lambda = " << lambdas(il) << ":\n";
    rhs.lambda = lambdas(il);
    Jac.lambda = lambdas(il);

    // loop over time step sizes
    for (int ih=0; ih<h.n_elem; ih++) {

      // create initial conditions
      mat y0_AB2(1,2);
      y0_AB2.col(0) = ytrue(t0);
      y0_AB2.col(1) = ytrue(t0-h(ih));

      // call stepper
      mat Y = AB2.Evolve(tspan, h(ih), y0_AB2);

      // output solution, errors, and overall error
      mat Yerr = abs(Y-Ytrue);
      e(ih) = Yerr.max();
      cout << "    h = " << h(ih) << "  steps = " << AB2.nsteps
           << "  max err = " << e(ih);
      if (ih > 0) {
        cout << "  conv rate = " << log(e(ih)/e(ih-1))/log(h(ih)/h(ih-1)) << endl;
      } else {
        cout << endl;
      }
    
    }
  }




  // Adams-Moulton-2
  cout << "\nAM-2 -- should be O(h^2):\n";
  vec AM2_a("1.0, -1.0");
  vec AM2_b("0.5, 0.5");
  LMMStepper AM2(rhs, Jac, y0, AM2_a, AM2_b);
  AM2.newt.tol = 1e-3;
  AM2.newt.maxit = 20;
  AM2.newt.show_iterates = false;

  // loop over lambda values
  for (int il=0; il<lambdas.n_elem; il++) {

    // set current lambda value into rhs and Jac objects
    cout << "  lambda = " << lambdas(il) << ":\n";
    rhs.lambda = lambdas(il);
    Jac.lambda = lambdas(il);

    // loop over time step sizes
    for (int ih=0; ih<h.n_elem; ih++) {

      // create initial conditions
      mat y0_AM2(1,1);
      y0_AM2.col(0) = ytrue(t0);

      // call stepper
      mat Y = AM2.Evolve(tspan, h(ih), y0_AM2);

      // output solution, errors, and overall error
      mat Yerr = abs(Y-Ytrue);
      e(ih) = Yerr.max();
      cout << "    h = " << h(ih) << "  steps = " << AM2.nsteps
           << "  NIters = " << AM2.nnewt << "  max err = " << e(ih);
      if (ih > 0) {
        cout << "  conv rate = " << log(e(ih)/e(ih-1))/log(h(ih)/h(ih-1)) << endl;
      } else {
        cout << endl;
      }
    
    }
  }





  // BDF-2
  cout << "\nBDF-2 -- should be O(h^2):\n";
  vec BDF2_a(3); BDF2_a(0) = 1.0; BDF2_a(1) = -4.0/3.0; BDF2_a(2) = 1.0/3.0;
  vec BDF2_b(3); BDF2_b.fill(0.0); BDF2_b(0) = 2.0/3.0;
  LMMStepper BDF2(rhs, Jac, y0, BDF2_a, BDF2_b);
  BDF2.newt.tol = 1e-3;
  BDF2.newt.maxit = 20;
  BDF2.newt.show_iterates = false;

  // loop over lambda values
  for (int il=0; il<lambdas.n_elem; il++) {

    // set current lambda value into rhs and Jac objects
    cout << "  lambda = " << lambdas(il) << ":\n";
    rhs.lambda = lambdas(il);
    Jac.lambda = lambdas(il);

    // loop over time step sizes
    for (int ih=0; ih<h.n_elem; ih++) {

      // create initial conditions
      mat y0_BDF2(1,2);
      y0_BDF2.col(0) = ytrue(t0);
      y0_BDF2.col(1) = ytrue(t0-h(ih));

      // call stepper
      mat Y = BDF2.Evolve(tspan, h(ih), y0_BDF2);

      // output solution, errors, and overall error
      mat Yerr = abs(Y-Ytrue);
      e(ih) = Yerr.max();
      cout << "    h = " << h(ih) << "  steps = " << BDF2.nsteps
           << "  NIters = " << BDF2.nnewt << "  max err = " << e(ih);
      if (ih > 0) {
        cout << "  conv rate = " << log(e(ih)/e(ih-1))/log(h(ih)/h(ih-1)) << endl;
      } else {
        cout << endl;
      }
    
    }
  }

  return 0;
}
