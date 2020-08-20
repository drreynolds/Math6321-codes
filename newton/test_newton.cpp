/* Main routine to test the Newton solver

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2016  */

#include <iostream>
#include <iomanip>
#include "resid.hpp"
#include "newton.hpp"

using namespace arma;

// Define classes to compute the nonlinear residual and its Jacobian

//    residual function class -- instantiates a ResidualFunction
class MyResid: public ResidualFunction {
public:
  double p;                              // stores some local data
  int Evaluate(vec& y, vec& r) {   // evaluate the residual, r(y)
    r(0) = cos(y(0))*cosh(y(0)) + p;
    r(1) = atan(y(1));
    return 0;
  }
};

//    residual Jacobian class -- instantiates a ResidualJacobian
class MyResidJac: public ResidualJacobian {
public:
  int Evaluate(vec& y, mat& J) {   // evaluate the residual Jacobian, J(y)
    J.fill(0.0);
    J(0,0) = cos(y(0))*sinh(y(0)) - sin(y(0))*cosh(y(0));
    J(1,1) = 1.0/(1.0+y(1)*y(1));
    return 0;
  }
};



// main routine
int main() {

  // create residual and Jacobian function objects
  MyResid r;
  MyResidJac J;
  r.p = 1.0;      // set the "p" value in the residual function

  // initial guess
  vec y0("1.5, 1.3");
  vec y(y0);

  // create Newton solver object
  double tol = 1.0;
  double rtol = 1.0e-5;
  double atol = 1.0e-12;
  vec w(2);
  w(0) = 1.0/(rtol*abs(y0(0)) + atol);
  w(1) = 1.0/(rtol*abs(y0(1)) + atol);
  int maxit = 100;
  bool show_iters = true;
  NewtonSolver newt(r, J, tol, w, maxit, y, show_iters);

  // call solver, output solution (overwrites y with the solution)
  std::cout << "\nCalling Newton solver with p = 1.0, relative tolerance of 1e-5:\n";
  if (newt.Solve(y) != 0) {
    std::cerr << "Newton failure\n";
  } else {
    std::cout << "Newton solution:  " << std::setprecision(12) 
              << y(0) << "  " << y(1) << endl;
    std::cout << "Required " << newt.GetIters() << " iterations\n";
    std::cout << "Final error norm = " << newt.GetErrorNorm() << "\n";
  }

  // update the problem and solver with new information
  r.p = -1.5;                           // update "p" in the residual function
  w(0) = 1.0/(1.e-7*abs(y0(0)) + atol); // update the solution relative tolerances
  w(1) = 1.0/(1.e-7*abs(y0(1)) + atol);
  y = y0;                               // reset the initial guess

  // call solver, output solution
  std::cout << "\nCalling Newton solver with p = -1.5, relative tolerance of 1e-7:\n";
  if (newt.Solve(y) != 0) {
    std::cerr << "Newton failure\n";
  } else {
    std::cout << "Newton solution:  " << std::setprecision(12) 
              << y(0) << "  " << y(1) << endl;
    std::cout << "Required " << newt.GetIters() << " iterations\n";
    std::cout << "Final error norm = " << newt.GetErrorNorm() << "\n";
  }

  return 0;
}
