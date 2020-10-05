/* Explicit 4th-order Runge-Kutta time stepper class header file.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#ifndef ERK4_DEFINED__
#define ERK4_DEFINED__

// Inclusions
#include <cmath>
#include "rhs.hpp"


// Explicit RK4 time stepper class
class ERK4Stepper {

 private:

  RHSFunction *frhs;           // pointer to ODE RHS function
  arma::vec k0, k1, k2, k3, z; // reused vectors
  arma::mat A;                 // Butcher table
  arma::vec b, c;      

 public:

  // number of steps in last call
  unsigned long int nsteps;

  // constructor (sets RHS function pointer, allocates local data)
  ERK4Stepper(RHSFunction& frhs_, arma::vec& y) {
    frhs = &frhs_;      // store RHSFunction pointer
    k0 = arma::vec(y);  // allocate reusable data
    k1 = arma::vec(y);
    k2 = arma::vec(y);
    k3 = arma::vec(y);
    z = arma::vec(y);   //   based on size of y
    nsteps = 0;
    A = arma::mat(4,4); A.fill(0.0);
    A(1,0) = 0.5;
    A(2,1) = 0.5;
    A(3,2) = 1.0;
    c = arma::vec(4);
    c(0) = 0.0;
    c(1) = 0.5;
    c(2) = 0.5;
    c(3) = 1.0;
    b = arma::vec(4);
    b(0) = 1.0/6.0;
    b(1) = 1.0/3.0;
    b(2) = 1.0/3.0;
    b(3) = 1.0/6.0;
  };

  // Evolve routine (evolves the solution)
  arma::mat Evolve(arma::vec tspan, double h, arma::vec y);

};

#endif
