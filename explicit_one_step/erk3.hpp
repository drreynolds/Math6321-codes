/* Explicit 3rd-order Runge-Kutta time stepper class header file.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#ifndef ERK3_DEFINED__
#define ERK3_DEFINED__

// Inclusions
#include <cmath>
#include "rhs.hpp"


// Explicit RK3 time stepper class
class ERK3Stepper {

 private:

  RHSFunction *frhs;       // pointer to ODE RHS function
  arma::vec k0, k1, k2, z; // reused vectors
  arma::mat A;             // Butcher table
  arma::vec b, c;      

 public:

  // number of steps in last call
  unsigned long int nsteps;

  // constructor (sets RHS function pointer, allocates local data)
  ERK3Stepper(RHSFunction& frhs_, arma::vec& y) {
    frhs = &frhs_;      // store RHSFunction pointer
    k0 = arma::vec(y);  // allocate reusable data
    k1 = arma::vec(y);  // allocate reusable data
    k2 = arma::vec(y);  // allocate reusable data
    z = arma::vec(y);   //   based on size of y
    nsteps = 0;
    A = arma::mat(3,3); A.fill(0.0);  // A = [0   0   0]
    A(1,0) = 2.0/3.0;                 //     [2/3 0   0]
    A(2,1) = 2.0/3.0;                 //     [0   2/3 0]
    c = arma::vec(3);
    c(0) = 0.0;
    c(1) = 2.0/3.0;
    c(2) = 2.0/3.0;
    b = arma::vec(3);
    b(0) = 0.25;
    b(1) = 3.0/8.0;
    b(2) = 3.0/8.0;
  };

  // Evolve routine (evolves the solution)
  arma::mat Evolve(arma::vec tspan, double h, arma::vec y);

};

#endif
