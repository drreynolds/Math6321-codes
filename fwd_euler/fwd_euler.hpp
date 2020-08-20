/* Forward Euler time stepper class header file.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020 */

#ifndef FORWARD_EULER_DEFINED__
#define FORWARD_EULER_DEFINED__

// Inclusions
#include <cmath>
#include "rhs.hpp"


// Forward Euler time stepper class
class ForwardEulerStepper {

 private:

  // private reusable local data
  arma::vec f;               // storage for ODE RHS vector
  RHSFunction *frhs;         // pointer to ODE RHS function

 public:

  // number of steps in last call
  unsigned long int nsteps;

  // constructor (sets RHS function pointer, copies y for local data)
  ForwardEulerStepper(RHSFunction& frhs_, arma::vec& y) {
    frhs = &frhs_;
    f = y;
    nsteps = 0;
  };

  // Evolve routine (evolves the solution via forward Euler)
  arma::mat Evolve(arma::vec tspan, double h, arma::vec y);

};

#endif
