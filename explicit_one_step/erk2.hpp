/* Explicit 2nd-order Runge-Kutta time stepper class header file.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#ifndef ERK2_DEFINED__
#define ERK2_DEFINED__

// Inclusions
#include <cmath>
#include "rhs.hpp"


// Explicit RK2 time stepper class
class ERK2Stepper {

 private:

  RHSFunction *frhs;   // pointer to ODE RHS function
  arma::vec k, z;      // reused vectors

 public:

  // number of steps in last call
  unsigned long int nsteps;

  // constructor (sets RHS function pointer, allocates local data)
  ERK2Stepper(RHSFunction& frhs_, arma::vec& y) {
    frhs = &frhs_;      // store RHSFunction pointer
    k = arma::vec(y);   // allocate reusable data
    z = arma::vec(y);   //   based on size of y
    nsteps = 0;
  };

  // Evolve routine (evolves the solution)
  arma::mat Evolve(arma::vec tspan, double h, arma::vec y);

};

#endif
