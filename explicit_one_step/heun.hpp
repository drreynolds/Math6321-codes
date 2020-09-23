/* Heun method (2nd-order explicit Runge-Kutta) time stepper class header file.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#ifndef HEUN_DEFINED__
#define HEUN_DEFINED__

// Inclusions
#include <cmath>
#include "rhs.hpp"


// Heun Explicit RK2 time stepper class
class HeunStepper {

 private:

  RHSFunction *frhs;    // pointer to ODE RHS function
  arma::vec k1, k2, z;  // reused vectors

 public:

  // number of steps in last call
  unsigned long int nsteps;

  // constructor (sets RHS function pointer, allocates local data)
  HeunStepper(RHSFunction& frhs_, arma::vec& y) {
    frhs = &frhs_;      // store RHSFunction pointer
    k1 = arma::vec(y);  // allocate reusable data
    k2 = arma::vec(y);  //   based on size of y
    z  = arma::vec(y);
    nsteps = 0;
  };

  // Evolve routine (evolves the solution)
  arma::mat Evolve(arma::vec tspan, double h, arma::vec y);

};

#endif
