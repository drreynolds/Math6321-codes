/* Newton nonlinear solver class implementation file.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#include <iomanip>
#include <cmath>
#include "newton.hpp"


using namespace arma;

// Newton solver construction routine
//
// Inputs:  fres_  -- the ResidualFunction to use
//          Jres_  -- the JacobianFunction to use
//          rtol_  -- the desired relative tolerance
//          atol_  -- the desired absolute tolerances
//          maxit_ -- the maximum allowed number of iterations
//          y      -- template solution vector (only used to clone)
NewtonSolver::NewtonSolver(ResidualFunction& fres_, ResidualJacobian& Jres_, 
                           const double tol_, const vec& w_, const int maxit_, 
                           const vec& y, const bool show_iterates_) {

  // set pointers to problem-defining function objects
  fres = &fres_;
  Jres = &Jres_;

  // set error weight vector pointer, tolerance
  tol = tol_;
  w = &w_;

  // set remaining solver parameters
  show_iterates = show_iterates_;
  maxit = maxit_;

  // create reusable solver objects (clone off of y)
  f = vec(y);
  s = vec(y);
  J = mat(y.size(), y.size());
  
  // initialize statistics
  iters = 0;
  error_norm = 0.0;

};

// Error-weight norm to be used for convergence tests:
//   max_i | w_i*e_i |
// where w is the error-weight vector stored in the NewtonSolver
// object, and e is the input vector.
double NewtonSolver::EWTNorm(const arma::vec& e) {
  double nrm = 0.0;
  for (size_t i=0; i<e.size(); i++) {
    double we = (*w)(i) * e(i);
    nrm = std::max(nrm, std::abs(we));
  }
  return nrm;
}

// The actual Newton solver routine
//
// Input:   y  -- the initial guess
// Outputs: y  -- the computed solution
//  
// The return value is one of:
//          0 => successful solve
//         -1 => bad function call or input
//          1 => non-convergent iteration
int NewtonSolver::Solve(vec& y) {

  // set initial residual value
  if (fres->Evaluate(y, f) != 0) {
    std::cerr << "NewtonSolver::Solve error: residual function failure\n";
    return -1;
  }

  // perform iterations
  for (iters=1; iters<=maxit; iters++) {

    // evaluate Jacobian 
    if (Jres->Evaluate(y, J) != 0) {
      std::cerr << "NewtonSolver::Solve error: Jacobian function failure\n";
      return -1;
    }

    // compute Newton update, norm
    if (arma::solve(s, J, f) == false) {
      std::cerr << "NewtonSolver::Solve error: linear solver failure\n";
      return -1;
    }
    error_norm = EWTNorm(s);

    // perform update
    y -= s;

    // update residual
    if (fres->Evaluate(y, f) != 0) {
      std::cerr << "NewtonSolver::Solve error: residual function failure\n";
      return -1;
    }

    // output convergence information
    if (show_iterates)
      printf("   iter %3i, ||s*w||_inf = %7.2e, ||f(x)*w||_inf = %7.2e\n",
             iters, error_norm, EWTNorm(f));

    // check for convergence, return if successful
    if (error_norm < tol)  return 0;

  }

  // if we've made it here, Newton did not converge, output warning and return
  std::cerr << "\nNewtonSolver::Solve WARNING: nonconvergence after " << maxit 
	    << " iterations (||s|| = " << error_norm << ")\n";
  return 1;
}

