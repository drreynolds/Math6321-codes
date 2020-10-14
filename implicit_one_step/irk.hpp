/* IRK time stepper class header file.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#ifndef IRK_DEFINED__
#define IRK_DEFINED__

// Inclusions
#include <cmath>
#include "rhs.hpp"
#include "newton.hpp"


// IRK residual function class -- implements an implicit
// Runge-Kutta-specific ResidualFunction to be supplied
// to the Newton solver.
class IRKResid: public ResidualFunction {
public:

  // data required to evaluate IRK nonlinear residual
  RHSFunction *frhs;  // pointer to ODE RHS function
  double t;           // current time
  double h;           // current step size
  arma::vec *yold;    // pointer to solution at old time step
  arma::mat *A;       // pointers to Butcher table components
  arma::vec *c;
  arma::vec zj;       // temporary storage
  arma::vec kj;       // temporary storage
  size_t s;           // num IRK stages
  size_t m;           // size of IVP system

  // constructor (sets RHS function, old solution vector pointers, Butcher table pointers)
  IRKResid(RHSFunction& frhs_, arma::vec& yold_,
           arma::mat& A_, arma::vec& c_) {
    frhs = &frhs_;  yold = &yold_;    // set pointers to inputs
    A = &A_;  c = &c_;
    m = yold_.n_elem;
    s = c_.n_elem;
    zj = arma::vec(m);
    kj = arma::vec(m);
  };

  // residual evaluation routine
  int Evaluate(arma::vec& z, arma::vec& resid);
};



// IRK residual Jacobian function class -- implements
// an implicit Runge-Kutta-specific ResidualJacobian to be
// supplied to the Newton solver.
class IRKResidJac: public ResidualJacobian {
public:

  // data required to evaluate IRK residual Jacobian
  RHSJacobian *Jrhs;       // ODE RHS Jacobian function pointer
  double t;                // current time
  double h;                // current step size
  arma::mat *A;            // pointers to Butcher table structures
  arma::vec *c;
  arma::vec zj;            // temporary storage
  arma::mat Jj;
  size_t s;                // num IRK stages
  size_t m;                // size of IVP system

  // constructor (sets RHS Jacobian function pointer)
  IRKResidJac(RHSJacobian& Jrhs_, arma::vec& yold_,
              arma::mat& A_, arma::vec& c_) {
    // set pointers to inputs
    Jrhs = &Jrhs_;
    A = &A_;  c = &c_;
    // store problem sizes
    m = yold_.n_elem;
    s = c_.n_elem;
    // create single-stage vector/Jacobian storage
    Jj = arma::mat(m,m);
    zj = arma::vec(m);
  };

  // residual Jacobian evaluation routine
  int Evaluate(arma::vec& z, arma::mat& J);
};



// IRK time stepper class
class IRKStepper {

 private:

  // private reusable local data
  RHSFunction *frhs;     // pointer to ODE RHS function
  arma::vec yold;        // old solution vector
  arma::vec w;           // error weight vector
  arma::mat A;           // Butcher table structures
  arma::vec b;
  arma::vec c;
  IRKResid r;            // IRK residual function
  IRKResidJac rJac;      // IRK residual Jacobian function
  arma::vec z;           // multi-stage vector
  arma::vec zj;          // temporary storage
  arma::vec kj;
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
    for (size_t j=0; j<s; j++)
      for (size_t i=0; i<m; i++)
        w(j*m + i) = 1.0 / (atol(i) + rtol * std::abs(y(i)));
  }

public:

  // IRK stepper construction routine (allocates local data)
  //
  // Inputs:  frhs   holds the RHSFunction to use
  //          Jrhs   holds the RHSJacobian to use
  //          y      holds an example solution vector (only used for cloning)
  //          A,b,c  Butcher table to use
  IRKStepper(RHSFunction& frhs_, RHSJacobian& Jrhs, arma::vec& y,
             arma::mat& A_, arma::vec& b_, arma::vec&c_)
  : s(c_.n_elem)                      // set stage count, problem size
  , m(y.n_elem)
  , frhs(&frhs_)
  , yold(arma::vec(y.n_elem))         // create local vectors
  , atol(arma::vec(y.n_elem))
  , w(arma::vec(c_.n_elem*y.n_elem))
  , z(arma::vec(c_.n_elem*y.n_elem))
  , zj(arma::vec(y.n_elem))
  , kj(arma::vec(y.n_elem))
  , A(A_)                             // copy Butcher table
  , b(b_)
  , c(c_)
  , r(IRKResid(frhs_,yold,A,c))       // construct nonlin. resid.
  , rJac(IRKResidJac(Jrhs,yold,A,c))  // construct nonlin. Jac.
  , rtol(1.0e-7)                      // default rtol value
  , nsteps(0)                         // initial counter values
  , nnewt(0)
  , newt(NewtonSolver(r, rJac, 1.0, w, 100, z, false))
  {
    // update atol and error weight values
    atol.fill(1.0e-11);     // absolute tolerances
    error_weight(y);
  };

  // ComputeResult routine (calculates ynew, given solution z to nonlinear system)
  int ComputeResult(double t, double h, arma::vec& z, arma::vec& y);

  // Evolve routine (evolves the solution via IRK method)
  arma::mat Evolve(arma::vec& tspan, double h, arma::vec y);

};

#endif
