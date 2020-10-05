/* Explicit 4th-order Runge-Kutta time stepper class implementation file.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#include "erk4.hpp"

using namespace std;
using namespace arma;

// The explicit 4th-order Runge-Kutta time step evolution routine
//
// Inputs:  tspan holds the time intervals, [t0, t1, ..., tN]
//          h holds the desired time step size
//          y holds the initial condition, y(t0)
// Outputs: the output matrix holds the computed solution at
//          all tspan values,
//            [y(t0), y(t1), ..., y(tN)]
mat ERK4Stepper::Evolve(vec tspan, double h, vec y) {

  // store sizes
  size_t m = y.n_elem;
  size_t N = tspan.n_elem-1;

  // initialize output
  mat Y(m, N+1, fill::zeros);
  Y.col(0) = y;

  // reset nsteps counter, current time value
  nsteps = 0;
  double t = tspan(0);

  // check for legal inputs
  if (h <= 0.0) {
    cerr << "ERK4Stepper: Illegal h\n";
    return Y;
  }
  for (size_t tstep=0; tstep<N; tstep++) {
    if (tspan(tstep+1) < tspan(tstep)) {
      cerr << "ERK4Stepper: Illegal tspan\n";
      return Y;
    }
  }

  // iterate over output time steps
  for (size_t tstep=0; tstep<N; tstep++) {

    // figure out how many time steps in this output interval
    size_t Nint = (tspan(tstep+1)-tspan(tstep)) / h;
    if ((tspan(tstep+1) - (tspan(tstep)+Nint*h)) > sqrt(eps(tspan(tstep+1))))  Nint++;

    // loop over internal steps to get to desired output time
    for (size_t i=0; i<Nint; i++) {

      // last step only: update h to stop directly at final time
      double hcur = h;
      if (i == Nint-1)  hcur = tspan(tstep+1)-t;

      //------- compute time-step update -------//

      // stage 0: compute k0 at old step solution
      if (frhs->Evaluate(t, y, k0) != 0) {
        std::cerr << "ERK4Stepper::Evolve: Error in ODE RHS function\n";
        return Y;
      }

      // stage 1: compute z1 (store in z) and corresponding k1
      z = y + (A(1,0)*hcur)*k0;
      if (frhs->Evaluate(t+c(1)*hcur, z, k1) != 0) {
        std::cerr << "ERK4Stepper::Evolve: Error in ODE RHS function\n";
        return Y;
      }

      // stage 2: compute z2 (store in z) and corresponding k2
      z = y + (A(2,1)*hcur)*k1;
      if (frhs->Evaluate(t+c(2)*hcur, z, k2) != 0) {
        std::cerr << "ERK4Stepper::Evolve: Error in ODE RHS function\n";
        return Y;
      }

      // stage 3: compute z3 (store in z) and corresponding k3
      z = y + (A(3,2)*hcur)*k2;
      if (frhs->Evaluate(t+c(3)*hcur, z, k3) != 0) {
        std::cerr << "ERK4Stepper::Evolve: Error in ODE RHS function\n";
        return Y;
      }

      // update solution
      y += hcur*(b(0)*k0 + b(1)*k1 + b(2)*k2 + b(3)*k3);

      //----------------------------------------//

      // update current time, nsteps counter
      t += hcur;
      nsteps++;

    }

    // store updated solution in output array
    Y.col(tstep+1) = y;

  }

  return Y;
}
