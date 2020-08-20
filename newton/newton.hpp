/* Newton nonlinear solver class header file.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#ifndef NEWTON_DEFINED__
#define NEWTON_DEFINED__

// Inclusions
#include "resid.hpp"


// Newton solver class
class NewtonSolver {

 private:

  // private reusable data
  arma::vec f;      // stores nonlinear residual vector
  arma::vec s;      // stores Newton update vector
  arma::mat J;      // stores nonlinear residual Jacobian matrix

  // private pointers to problem-defining function objects
  ResidualFunction *fres;   // nonlinear residual function pointer
  ResidualJacobian *Jres;   // nonlinear residual Jacobian function pointer

  // private solver parameters
  const arma::vec *w;       // pointer to desired error weight vector

  // private statistics 
  int iters;                // iteration counter (reset in each solve)
  double error_norm;        // most recent error estimate (in error-weight max norm)

 public:

  // public solver parameters
  double tol;               // desired tolerance (in error-weight max norm)
  int maxit;                // maximum desired Newton iterations
  bool show_iterates;       // flag to output iteration information

  // Constructor
  NewtonSolver(ResidualFunction& fres_, ResidualJacobian& Jres_, 
	            const double tol_, const arma::vec& w_, const int maxit_, 
               const arma::vec& y, const bool show_iterates_);

  // Newton solver routine
  int Solve(arma::vec& y);

  // Error-weight max norm utility routine
  double EWTNorm(const arma::vec& e);

  // Parameter update & statistics accessor routines
  void ResetIters() { iters = 0; };
  const int GetIters() { return iters; };
  const double GetErrorNorm() { return error_norm; };

};

#endif
