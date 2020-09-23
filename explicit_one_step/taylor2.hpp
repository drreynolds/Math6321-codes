/* Explicit 2nd-order Taylor series method time stepper class header file.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#ifndef TAYLOR2_DEFINED__
#define TAYLOR2_DEFINED__

// Inclusions
#include <cmath>
#include "rhs.hpp"


// Explicit 2nd-order Taylor series method time stepper class
class Taylor2Stepper {

 private:

  RHSFunction *frhs;      // pointer to ODE RHS function
  RHSFunction *frhs_t;    // pointer to ODE RHS function, time derivative
  RHSJacobian *frhs_y;    // pointer to ODE RHS function, y derivative
  arma::vec f, ft, fyf;   // reused vectors
  arma::mat fy;           // reused matrix

 public:

  // number of steps in last call
  unsigned long int nsteps;

  // constructor (sets RHS function pointer)
  Taylor2Stepper(RHSFunction& frhs_, RHSFunction& frhs_t_,
                 RHSJacobian& frhs_y_, arma::vec& y) {
    frhs   = &frhs_;      // store RHSFunction pointers
    frhs_t = &frhs_t_;
    frhs_y = &frhs_y_;
    f   = arma::vec(y);   // allocate reusable data
    ft  = arma::vec(y);   //   based on size of y
    fyf = arma::vec(y);
    fy  = arma::mat(y.n_elem, y.n_elem);
    nsteps = 0;
  };

  // Evolve routine (evolves the solution)
  arma::mat Evolve(arma::vec tspan, double h, arma::vec y);

};

#endif
