/* DIRK time stepper class header file.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#ifndef DIRK_DEFINED__
#define DIRK_DEFINED__

// Inclusions
#include <cmath>
#include "rhs.hpp"
#include "newton.hpp"


// DIRK residual function class -- implements an implicit 
// Runge-Kutta-specific ResidualFunction to be supplied 
// to the Newton solver.
class DIRKResid: public ResidualFunction {
public:

  // data required to evaluate DIRK nonlinear residual
  RHSFunction *frhs;  // pointer to ODE RHS function
  double t;           // current time
  double h;           // current step size
  arma::vec a;        // vector of previous stage 'data'
  arma::mat A;        // Butcher table components
  arma::vec c;
  size_t s;           // num DIRK stages
  size_t m;           // size of IVP system
  int cur_stage;      // current stage index

  // constructor (sets RHS function, old solution vector pointers, Butcher table pointers)
  DIRKResid(RHSFunction& frhs_, arma::vec& y, 
            arma::mat& A_, arma::vec& c_) {
    frhs = &frhs_;
    a = arma::vec(m);
    A = A_;  
    c = c_;
    s = c_.n_elem;
    m = y.n_elem;
    cur_stage = 0;
    t = NAN;
    h = NAN;
  };

  // residual evaluation routine
  int Evaluate(arma::vec& z, arma::vec& resid);
};



// DIRK residual Jacobian function class -- implements 
// an implicit Runge-Kutta-specific ResidualJacobian to be 
// supplied to the Newton solver.
class DIRKResidJac: public ResidualJacobian {
public:

  // data required to evaluate IRK residual Jacobian
  RHSJacobian *Jrhs;       // ODE RHS Jacobian function pointer
  double t;                // current time
  double h;                // current step size
  arma::mat A;             // Butcher table structures
  arma::vec c;
  size_t s;                // num IRK stages
  size_t m;                // size of IVP system
  int cur_stage;           // current stage index

  // constructor (sets RHS Jacobian function pointer)
  DIRKResidJac(RHSJacobian& Jrhs_, arma::vec& y, 
               arma::mat& A_, arma::vec& c_) { 
    Jrhs = &Jrhs_;        
    A = A_;  
    c = c_;
    s = c_.n_elem;
    m = y.n_elem;
    cur_stage = 0;
    t = NAN;
    h = NAN;
  };

  // residual Jacobian evaluation routine
  int Evaluate(arma::vec& z, arma::mat& J);
};



// DIRK time stepper class
class DIRKStepper {

 private:

  // private reusable local data
  RHSFunction *frhs;     // pointer to ODE RHS function
  arma::vec yold;        // old solution vector
  arma::vec w;           // error weight vector
  arma::mat A;           // Butcher table structures
  arma::vec b;
  arma::vec c;
  DIRKResid r;           // DIRK residual function
  DIRKResidJac rJac;     // DIRK residual Jacobian function
  arma::vec z;           // current stage vector
  arma::vec k;           // current RHS vector
  arma::mat K;           // storage for all stage RHS vectors
  size_t s;              // num IRK stages
  size_t m;              // size of IVP system

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
    for (size_t i=0; i<m; i++)
      w(i) = 1.0 / (atol(i) + rtol * std::abs(y(i)));
  }

public:

  // DIRK stepper construction routine (allocates local data)
  //
  // Inputs:  frhs   holds the RHSFunction to use
  //          Jrhs   holds the RHSJacobian to use
  //          y      holds an example solution vector (only used for cloning)
  //          A,b,c  Butcher table to use
  DIRKStepper(RHSFunction& frhs_, RHSJacobian& Jrhs, arma::vec& y, 
              arma::mat& A_, arma::vec& b_, arma::vec&c_)
  : s(c_.n_elem)                       // set stage count, problem size
  , m(y.n_elem)
  , frhs(&frhs_)
  , yold(arma::vec(y.n_elem))          // create local vectors
  , atol(arma::vec(y.n_elem))
  , w(arma::vec(y.n_elem))
  , z(arma::vec(y.n_elem))
  , k(arma::vec(y.n_elem))
  , K(arma::mat(y.n_elem,c_.n_elem))
  , A(A_)                              // copy Butcher table
  , b(b_)
  , c(c_)
  , r(DIRKResid(frhs_,yold,A,c))       // construct nonlin. resid.
  , rJac(DIRKResidJac(Jrhs,yold,A,c))  // construct nonlin. Jac.
  , rtol(1.0e-7)                       // default rtol value
  , nsteps(0)                          // initial counter values
  , nnewt(0)
  , newt(NewtonSolver(r, rJac, 1.0, w, 100, z, false))
  {
    // update atol and error weight values
    atol.fill(1.0e-11);     // absolute tolerances
    error_weight(y);
  };

  // Evolve routine (evolves the solution via DIRK method)
  arma::mat Evolve(arma::vec& tspan, double h, arma::vec y);

};

#endif
