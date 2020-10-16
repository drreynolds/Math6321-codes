/* Newton nonlinear solver class header/implementation file.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#ifndef NEWTON_DEFINED__
#define NEWTON_DEFINED__

// Inclusions
#include <cmath>
#include <armadillo>
#include <iomanip>


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
  //
  // Inputs:  fres_  -- the ResidualFunction to use
  //          Jres_  -- the JacobianFunction to use
  //          tol_   -- the desired solution tolerance
  //          w_     -- the error weight vector to use
  //          maxit_ -- the maximum allowed number of iterations
  //          y      -- template solution vector (only used to clone)
  //          show_iterates_ -- enable/disable printing iterate info
  NewtonSolver(ResidualFunction& fres_, ResidualJacobian& Jres_,
               const double tol_, const arma::vec& w_, const int maxit_,
               const arma::vec& y, const bool show_iterates_) {

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
    f = arma::vec(y);
    s = arma::vec(y);
    J = arma::mat(y.size(), y.size());

    // initialize statistics
    iters = 0;
    error_norm = 0.0;
  }

  // Utility routine to ensure that the Newton solver object has the current
  // fres, Jres and w pointers
  void UpdatePointers(ResidualFunction& fres_,
                      ResidualJacobian& Jres_,
                      const arma::vec& w_) {
    fres = &fres_;
    Jres = &Jres_;
    w = &w_;
  };

  // Error-weight max norm utility routine for convergence tests
  //   max_i | w_i*e_i |
  // where w is the error-weight vector stored in the NewtonSolver
  // object, and e is the input vector.
  double EWTNorm(const arma::vec& e) {
    double nrm = 0.0;
    for (size_t i=0; i<e.size(); i++) {
      double we = (*w)(i) * e(i);
      nrm = std::max(nrm, std::abs(we));
    }
    return nrm;
  }

  // Newton solver routine
  //
  // Input:   y  -- the initial guess
  // Outputs: y  -- the computed solution
  //
  // The return value is one of:
  //          0 => successful solve
  //         -1 => bad function call or input
  //          1 => non-convergent iteration
  int Solve(arma::vec& y) {

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

  // Parameter update & statistics accessor routines
  void ResetIters() { iters = 0; };
  const int GetIters() { return iters; };
  const double GetErrorNorm() { return error_norm; };

};

#endif
