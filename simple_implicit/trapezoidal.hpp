/* Trapezoidal time stepper class header file.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#ifndef TRAPEZOIDAL_DEFINED__
#define TRAPEZOIDAL_DEFINED__

// Inclusions
#include <cmath>
#include "rhs.hpp"
#include "newton.hpp"


// Trapezoidal residual function class -- implements a
// trapezoidal-method-specific ResidualFunction to be supplied
// to the Newton solver.
class TrapResid: public ResidualFunction {
public:

  // data required to evaluate trapezoidal method nonlinear residual
  RHSFunction *frhs;  // pointer to ODE RHS function
  double t;           // current time
  double h;           // current step size
  arma::vec *yold;    // pointer to solution at old time step
  arma::vec fold;     // storage for f at old solution

  // constructor (sets RHSFunction and old solution pointers)
  TrapResid(RHSFunction& frhs_, arma::vec& yold_) {
    frhs = &frhs_;  yold = &yold_;
    fold = arma::vec(yold_);
  };

  // residual evaluation routine
  int Evaluate(arma::vec& y, arma::vec& resid) {

    // evaluate RHS function (store in resid)
    int ierr = frhs->Evaluate(t+h, y, resid);
    if (ierr != 0) {
      std::cerr << "Error in ODE RHS function = " << ierr << "\n";
      return ierr;
    }

    // combine pieces to fill residual, y-yold-h/2*(f(t+h,y)+f(t,yold))
    resid = y - (*yold) - 0.5*h*resid - 0.5*h*fold;

    // return success
    return 0;
  }
};



// Trapezoidal residual Jacobian function class -- implements
// a trapezoidal-method-specific ResidualJacobian to be supplied
// to the Newton solver.
class TrapResidJac: public ResidualJacobian {
public:

  // data required to evaluate trapezoidal method residual Jacobian
  RHSJacobian *Jrhs;   // ODE RHS Jacobian function pointer
  double t;            // current time
  double h;            // current step size

  // constructor (sets RHS Jacobian function pointer)
  TrapResidJac(RHSJacobian &Jrhs_) { Jrhs = &Jrhs_; };

  // Residual Jacobian evaluation routine
  int Evaluate(arma::vec& y, arma::mat& J) {

    // evaluate RHS function Jacobian, Jrhs (store in J)
    int ierr = Jrhs->Evaluate(t+h, y, J);
    if (ierr != 0) {
      std::cerr << "Error in ODE RHS Jacobian function = " << ierr << "\n";
      return ierr;
    }
    // combine pieces to fill residual Jacobian,  J = I - h*Jrhs
    J *= (-0.5*h);
    for (int i=0; i<J.n_rows; i++)
      J(i,i) += 1.0;

    // return success
    return 0;
  }
};



// Trapezoidal time stepper class
class TrapezoidalStepper {

private:

  // private reusable local data
  RHSFunction *frhs;     // pointer to ODE RHS function
  arma::vec yold;        // old solution vector
  arma::vec w;           // error weight vector
  TrapResid resid;       // trapezoidal method residual function pointer
  TrapResidJac residJac; // trapezoidal method residual Jacobian function pointer

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
    for (size_t i=0; i<y.size(); i++)
      w(i) = 1.0 / (atol(i) + rtol * std::abs(y(i)));
  }

public:

  // Trapezoidal method stepper construction routine (allocates local data)
  //
  // Inputs:  frhs_  holds the RHSFunction to use
  //          Jrhs_  holds the RHSJacobian to use
  //          y      holds an example solution vector (only used for cloning)
  TrapezoidalStepper(RHSFunction& frhs_, RHSJacobian& Jrhs_, arma::vec& y)
    : frhs(&frhs_)                   // point to ODE RHS function
    , yold(arma::vec(y))             // clone y to create yold
    , atol(arma::vec(y))             // clone y to create atol
    , w(arma::vec(y))                // clone y to create w
    , resid(TrapResid(frhs_, yold))  // construct nonlinear residual object
    , residJac(TrapResidJac(Jrhs_))  // construct nonlinear Jacobian object
    , rtol(1.0e-7)                   // default rtol value
    , nsteps(0)                      // initial counter values
    , nnewt(0)
    , newt(NewtonSolver(resid, residJac, 1.0, w, 100, y, false))
  {
    // update atol and error weight values
    atol.fill(1.0e-11);     // absolute tolerances
    error_weight(y);
  };

  // Evolve routine (evolves the solution via trapezoidal method)
  arma::mat Evolve(arma::vec& tspan, double h, arma::vec y);

};

#endif
