/* Backward Euler time stepper class header file.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#ifndef BACKWARD_EULER_DEFINED__
#define BACKWARD_EULER_DEFINED__

// Inclusions
#include <cmath>
#include "rhs.hpp"
#include "newton.hpp"


// Backward Euler residual function class -- implements a
// backward-Euler-specific ResidualFunction to be supplied
// to the Newton solver.
class BEResid: public ResidualFunction {
public:

  // data required to evaluate backward Euler nonlinear residual
  RHSFunction *frhs;  // pointer to ODE RHS function
  double t;           // current time
  double h;           // current step size
  arma::vec *yold;    // pointer to solution at old time step

  // constructor (sets RHSFunction and old solution pointers)
  BEResid(RHSFunction& frhs_, arma::vec& yold_) {
    frhs = &frhs_;  yold = &yold_;
  };

  // residual evaluation routine
  int Evaluate(arma::vec& y, arma::vec& resid) {

    // evaluate RHS function (store in resid)
    int ierr = frhs->Evaluate(t+h, y, resid);
    if (ierr != 0) {
      std::cerr << "Error in ODE RHS function = " << ierr << "\n";
      return ierr;
    }

    // combine pieces to fill residual, y-yold-h*f(t+h,y)
    resid = y - (*yold) - h*resid;

    // return success
    return 0;
  }
};



// Backward Euler residual Jacobian function class -- implements
// a backward-Euler-specific ResidualJacobian to be supplied
// to the Newton solver.
class BEResidJac: public ResidualJacobian {
public:

  // data required to evaluate backward Euler residual Jacobian
  RHSJacobian *Jrhs;   // ODE RHS Jacobian function pointer
  double t;            // current time
  double h;            // current step size

  // constructor (sets RHS Jacobian function pointer)
  BEResidJac(RHSJacobian &Jrhs_) { Jrhs = &Jrhs_; };

  // Residual Jacobian evaluation routine
  int Evaluate(arma::vec& y, arma::mat& J) {

    // evaluate RHS function Jacobian, Jrhs (store in J)
    int ierr = Jrhs->Evaluate(t+h, y, J);
    if (ierr != 0) {
      std::cerr << "Error in ODE RHS Jacobian function = " << ierr << "\n";
      return ierr;
    }
    // combine pieces to fill residual Jacobian,  J = I - h*Jrhs
    J *= (-h);
    for (int i=0; i<J.n_rows; i++)
      J(i,i) += 1.0;

    // return success
    return 0;
  }
};



// Backward Euler time stepper class
class BackwardEulerStepper {

private:

  // private reusable local data
  arma::vec yold;        // old solution vector
  arma::vec w;           // error weight vector
  BEResid resid;         // backward Euler residual function pointer
  BEResidJac residJac;   // backward Euler residual Jacobian function pointer

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

  // Backward Euler stepper construction routine (allocates local data)
  //
  // Inputs:  frhs_  holds the RHSFunction to use
  //          Jrhs_  holds the RHSJacobian to use
  //          y      holds an example solution vector (only used for cloning)
  BackwardEulerStepper(RHSFunction& frhs_, RHSJacobian& Jrhs_, arma::vec& y)
    : yold(arma::vec(y))           // clone y to create yold
    , atol(arma::vec(y))           // clone y to create atol
    , w(arma::vec(y))              // clone y to create w
    , resid(BEResid(frhs_, yold))  // construct nonlinear residual object
    , residJac(BEResidJac(Jrhs_))  // construct nonlinear Jacobian object
    , rtol(1.0e-7)                 // default rtol value
    , nsteps(0)                    // initial counter values
    , nnewt(0)
    , newt(NewtonSolver(resid, residJac, 1.0, w, 100, y, false))
  {
    // update atol and error weight values
    atol.fill(1.0e-11);     // absolute tolerances
    error_weight(y);
  };

  // Evolve routine (evolves the solution via backward Euler)
  arma::mat Evolve(arma::vec& tspan, double h, arma::vec y);

};

#endif
