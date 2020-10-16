/* Linear multistep time stepper class header file.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#ifndef LMM_DEFINED__
#define LMM_DEFINED__

// Inclusions
#include <cmath>
#include "rhs.hpp"
#include "newton.hpp"


// Linear multistep residual function class -- implements a
// LMM-specific ResidualFunction to be supplied to the Newton solver.
class LMMResid: public ResidualFunction {
public:

  // data required to evaluate LMM nonlinear residual
  RHSFunction *frhs;       // pointer to ODE RHS function
  double t;                // current time
  double h;                // current step size
  arma::mat *yold;         // matrix of old solution vectors
  arma::mat *fold;         // matrix of old right-hand side vectors
  arma::vec a;             // vector of LMM "a" coefficients
  arma::vec b;             // vector of LMM "b" coefficients
  arma::vec yn;            // scratch space for evaluating frhs
  arma::vec fn;

  // constructor (sets RHS function and old solution vector pointers)
  LMMResid(RHSFunction& frhs_, arma::vec& y, arma::vec& a_, arma::vec& b_)
    : frhs(&frhs_)
    , t(NAN)
    , h(NAN)
    , yold(NULL)
    , fold(NULL)
    , a(a_)
    , b(b_)
    , yn(arma::vec(y.n_elem))
    , fn(arma::vec(y.n_elem))
  {};

  // utility routine to update residual for a new solve
  void Update(arma::mat& yold_, arma::mat& fold_, double t_, double h_) {
    yold = &yold_;
    fold = &fold_;
    t = t_;
    h = h_;
  }

  // residual evaluation routine
  int Evaluate(arma::vec& y, arma::vec& resid);
};


// Linear multistep residual Jacobian function class -- implements
// a LMM-specific ResidualJacobian to be supplied to the Newton solver.
class LMMResidJac: public ResidualJacobian {
public:

  // data required to evaluate LMM residual Jacobian
  RHSJacobian *Jrhs;   // ODE RHS Jacobian function pointer
  double t;            // current time
  double h;            // current step size
  double b0;           // b0 coefficient

  // constructor (sets RHS Jacobian function pointer)
  LMMResidJac(RHSJacobian& Jrhs_, double b0_)
    : Jrhs(&Jrhs_)
    , b0(b0_) {};

  // utility routine to update Jacobian for a new solve
  void Update(double t_, double h_) {
    t = t_;
    h = h_;
  }

  // residual Jacobian evaluation routine
  int Evaluate(arma::vec& y, arma::mat& J);
};



// LMM time stepper class
class LMMStepper {

 private:

  // private reusable local data
  RHSFunction *frhs;      // pointer to ODE RHS function
  arma::mat yold;         // matrix of old solution vectors
  arma::mat fold;         // matrix of old right-hand side vectors
  arma::vec yn;           // storage for 'current' solution vector
  arma::vec w;            // error weight vector
  arma::vec a;            // LMM coefficients
  arma::vec b;
  LMMResid r;             // LMM residual function
  LMMResidJac rJac;       // LMM residual Jacobian function

 public:

  // nonlinear solver residual/absolute tolerances
  double rtol;
  arma::vec atol;

  // number of steps in last call
  unsigned long int nsteps;

  // total number of Newton iterations in last call
  unsigned long int nnewt;

  // Newton nonlinear solver pointer -- users can directly access/set
  // solver parameters:
  //   newt.tol
  //   newt.maxit
  //   newt.show_iterates
  NewtonSolver newt;

private:

  // utility routine to update the error weight vector
  void error_weight(arma::vec& y) {
    for (size_t i=0; i<y.n_elem; i++)
      w(i) = 1.0 / (atol(i) + rtol * std::abs(y(i)));
  }

  // utility routine to set up internal multistep structures
  int Initialize(double t0, double h, arma::mat& y0);

  // utility routine to update internal multistep structures
  int UpdateHistory(double tnew, arma::vec& ynew);

public:

  // LMM constructor routine (allocates local data)
  LMMStepper(RHSFunction& frhs_, RHSJacobian& Jrhs_, arma::vec& y,
             arma::vec& a_, arma::vec& b_)
  : frhs(&frhs_)
  , yold(arma::mat(y.n_elem,a_.n_elem-1)) // create local matrices/vectors
  , fold(arma::mat(y.n_elem,a_.n_elem-1))
  , yn(arma::vec(y.n_elem))
  , w(arma::vec(y.n_elem))
  , atol(arma::vec(y.n_elem))
  , a(a_)                                 // copy LMM coefficients
  , b(b_)
  , r(LMMResid(frhs_,yn,a,b))             // construct nonlin. resid.
  , rJac(LMMResidJac(Jrhs_,b(0)))         // construct nonlin. Jac.
  , rtol(1.0e-7)                          // default rtol value
  , nsteps(0)                             // initial counter values
  , nnewt(0)
  , newt(NewtonSolver(r, rJac, 1.0, w, 100, yn, false))
  {
    // update atol and error weight values
    atol.fill(1.0e-11);          // absolute tolerances
    error_weight(y);
  };

  // Evolve routine (evolves the solution via LMM)
  arma::mat Evolve(arma::vec& tspan, double h, arma::mat& y0);

};

#endif
