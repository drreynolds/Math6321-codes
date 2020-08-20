/* Newton nonlinear solver class header file.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#ifndef RESIDUAL_DEFINED__
#define RESIDUAL_DEFINED__

// Inclusions
#include <cmath>
#include <armadillo>


// Declare abstract base classes for residual and Jacobian to 
// define what the Newton solver expects from each.

//   Residual function abstract base class; derived classes 
//   must at least implement the Evaluate() routine
class ResidualFunction {
 public: 
  virtual int Evaluate(arma::vec& y, arma::vec& r) = 0;
};

//   Residual Jacobian function abstract base class; derived classes 
//   must at least implement the Evaluate() routine
class ResidualJacobian {
 public: 
  virtual int Evaluate(arma::vec& y, arma::mat& J) = 0;
};

#endif
