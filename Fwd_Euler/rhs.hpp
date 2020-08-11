/* ODE RHS and Jacobian function abstract base class definitions.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020 */

#ifndef ODE_RHS_DEFINED__
#define ODE_RHS_DEFINED__

// Inclusions
#include <armadillo>


// Declare abstract base classes for ODE RHS and its Jacobian, to
// define what the time integrator expects from each.

//   ODE RHS function abstract base class; derived classes
//   must at least implement the Evaluate() routine
class RHSFunction {
 public:
  virtual int Evaluate(double t, arma::vec& y, arma::vec& f) = 0;
};

//   ODE RHS Jacobian function abstract base class; derived
//   classes must at least implement the Evaluate() routine
class RHSJacobian {
 public:
  virtual int Evaluate(double t, arma::vec& y, arma::mat& J) = 0;
};

#endif
