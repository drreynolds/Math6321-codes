/* Main routine to test a set of linear multistep methods on the
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
  vec lambdas("-1.0, -10.0, -50.0, -1000.0");

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

  ////////// create linear multistep methods //////////
  string method_names[9];
  int initial_conditions[9];
  bool implicit[9];
  int imethod=0;

  // Adams-Bashforth-2
  method_names[imethod] = "AB-2";
  vec AB2_a("1.0, -1.0, 0.0");
  vec AB2_b("0.0, 1.5, -0.5");
  initial_conditions[imethod] = 2;
  implicit[imethod++] = false;
  LMMStepper AB2(rhs, Jac, y0, AB2_a, AB2_b);

  // Adams-Bashforth-3
  method_names[imethod] = "AB-3";
  vec AB3_a("1.0, -1.0, 0.0, 0.0");
  vec AB3_b(4);
  AB3_b(0) = 0.0;
  AB3_b(1) = 23.0/12.0;
  AB3_b(2) = -16.0/12.0;
  AB3_b(3) = 5.0/12.0;
  initial_conditions[imethod] = 3;
  implicit[imethod++] = false;
  LMMStepper AB3(rhs, Jac, y0, AB3_a, AB3_b);

  // Adams-Bashforth-4
  method_names[imethod] = "AB-4";
  vec AB4_a("1.0, -1.0, 0.0, 0.0, 0.0");
  vec AB4_b(5);
  AB4_b(0) = 0.0;
  AB4_b(1) = 55.0/24.0;
  AB4_b(2) = -59.0/24.0;
  AB4_b(3) = 37.0/24.0;
  AB4_b(4) = -9.0/24.0;
  initial_conditions[imethod] = 4;
  implicit[imethod++] = false;
  LMMStepper AB4(rhs, Jac, y0, AB4_a, AB4_b);

  // Adams-Moulton-2
  method_names[imethod] = "AM-2";
  vec AM2_a("1.0, -1.0");
  vec AM2_b("0.5, 0.5");
  initial_conditions[imethod] = 1;
  implicit[imethod++] = true;
  LMMStepper AM2(rhs, Jac, y0, AM2_a, AM2_b);

  // Adams-Moulton-3
  method_names[imethod] = "AM-3";
  vec AM3_a("1.0, -1.0, 0.0");
  vec AM3_b(3);
  AM3_b(0) = 5.0/12.0;
  AM3_b(1) = 8.0/12.0;
  AM3_b(2) = -1.0/12.0;
  initial_conditions[imethod] = 2;
  implicit[imethod++] = true;
  LMMStepper AM3(rhs, Jac, y0, AM3_a, AM3_b);

  // Adams-Moulton-4
  method_names[imethod] = "AM-4";
  vec AM4_a("1.0, -1.0, 0.0, 0.0");
  vec AM4_b(4);
  AM4_b(0) = 9.0/24.0;
  AM4_b(1) = 19.0/24.0;
  AM4_b(2) = -5.0/24.0;
  AM4_b(3) = 1.0/24.0;
  initial_conditions[imethod] = 3;
  implicit[imethod++] = true;
  LMMStepper AM4(rhs, Jac, y0, AM4_a, AM4_b);

  // BDF-2
  method_names[imethod] = "BDF-2";
  vec BDF2_a(3), BDF2_b(3);
  BDF2_a(0) = 1.0;
  BDF2_a(1) = -4.0/3.0;
  BDF2_a(2) = 1.0/3.0;
  BDF2_b.fill(0.0);
  BDF2_b(0) = 2.0/3.0;
  initial_conditions[imethod] = 2;
  implicit[imethod++] = true;
  LMMStepper BDF2(rhs, Jac, y0, BDF2_a, BDF2_b);

  // BDF-3
  method_names[imethod] = "BDF-3";
  vec BDF3_a(4), BDF3_b(4);
  BDF3_a(0) = 1.0;
  BDF3_a(1) = -18.0/11.0;
  BDF3_a(2) = 9.0/11.0;
  BDF3_a(3) = -2.0/11.0;
  BDF3_b.fill(0.0);
  BDF3_b(0) = 6.0/11.0;
  initial_conditions[imethod] = 3;
  implicit[imethod++] = true;
  LMMStepper BDF3(rhs, Jac, y0, BDF3_a, BDF3_b);

  // BDF-4
  method_names[imethod] = "BDF-4";
  vec BDF4_a(5), BDF4_b(5);
  BDF4_a(0) = 1.0;
  BDF4_a(1) = -48.0/25.0;
  BDF4_a(2) = 36.0/25.0;
  BDF4_a(3) = -16.0/25.0;
  BDF4_a(4) = 3.0/25.0;
  BDF4_b.fill(0.0);
  BDF4_b(0) = 12.0/25.0;
  initial_conditions[imethod] = 4;
  implicit[imethod++] = true;
  LMMStepper BDF4(rhs, Jac, y0, BDF4_a, BDF4_b);

  // combine method objects into an array
  LMMStepper methods[9] = {AB2, AB3, AB4, AM2, AM3, AM4, BDF2, BDF3, BDF4};


  // loop over methods
  for (imethod = 0; imethod<9; imethod++) {

    // set shortcut variables for this method
    string mname = method_names[imethod];
    int k = initial_conditions[imethod];
    bool impl = implicit[imethod];

    // output method name
    cout << endl << mname << ":\n";

    // update solver parameters for implicit methods
    if (implicit[imethod]) {
      methods[imethod].newt.tol = 1.e-3;
      methods[imethod].newt.maxit = 20;
      methods[imethod].newt.show_iterates = false;
    }

    // loop over lambda values
    for (int il=0; il<lambdas.n_elem; il++) {

      // set current lambda value into rhs and Jac objects
      cout << "  lambda = " << lambdas(il) << ":\n";
      rhs.lambda = lambdas(il);
      Jac.lambda = lambdas(il);

      // loop over time step sizes
      for (int ih=0; ih<h.n_elem; ih++) {

        // set initial conditions (backward in time)
        mat y0_LMM(1,k);
        for (int iic=0; iic<k; iic++)
          y0_LMM.col(iic) = ytrue(t0-iic*h(ih));

        // call stepper
        mat Y = methods[imethod].Evolve(tspan, h(ih), y0_LMM);

        // output solution, errors, and overall error
        mat Yerr = abs(Y-Ytrue);
        e(ih) = Yerr.max();
        if (implicit[imethod]) {
          cout << "    h = " << setw(6) << h(ih)
               << "   steps = " << setw(5) << methods[imethod].nsteps
               << "   NIters = " << setw(5) << methods[imethod].nnewt
               << "   max err = " << setw(10) << e(ih);
        } else {
          cout << "    h = " << setw(6) << h(ih)
               << "   steps = " << setw(5) << methods[imethod].nsteps
               << "   max err = " << setw(10) << e(ih);
        }
        if (ih > 0) {
          cout << "   conv rate = " << log(e(ih)/e(ih-1))/log(h(ih)/h(ih-1)) << endl;
        } else {
          cout << endl;
        }

      }
    }
  }

  return 0;
}
