/* Adaptive forward Euler time stepper class header file.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#ifndef ADAPT_EULER_DEFINED__
#define ADAPT_EULER_DEFINED__

// Inclusions
#include <cmath>
#include "rhs.hpp"


// Adaptive forward Euler time stepper class
class AdaptEuler {

 private:

  // private data
  RHSFunction *frhs;  // pointer to ODE RHS function
  arma::vec fn;       // local vector storage
  arma::vec y1;
  arma::vec y2;
  arma::vec yerr;
  double grow;        // maximum step size growth factor
  double safe;        // safety factor for step size estimate
  double fail;        // failed step reduction factor
  double ONEMSM;      // safety factors for
  double ONEPSM;      //    floating-point comparisons
  double alpha;       // exponent relating step to error

 public:

  // publicly-accesible input parameters
  double rtol;        // desired relative solution error
  double atol;        // desired absolute solution error
  long int maxit;     // maximum allowed number of steps

  // outputs
  double error_norm;  // current estimate of the local error ratio
  double h;           // current time step size
  long int fails;     // number of failed steps
  long int steps;     // number of successful steps

  // constructor (sets RHS function pointer & solver parameters, copies y for local data)
  AdaptEuler(RHSFunction& frhs_, double rtol_, double atol_, arma::vec& y);

  // Evolve routine (evolves the solution via adaptive forward Euler)
  arma::mat Evolve(arma::vec tspan, arma::vec y);

};

#endif
