/* Explicit 2nd-order Runge-Kutta time stepper class implementation file.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#include "erk2.hpp"

using namespace std;
using namespace arma;

// The explicit 2nd-order Runge-Kutta time step evolution routine
//
// Inputs:  tspan holds the time intervals, [t0, t1, ..., tN]
//          h holds the desired time step size
//          y holds the initial condition, y(t0)
// Outputs: the output matrix holds the computed solution at
//          all tspan values,
//            [y(t0), y(t1), ..., y(tN)]
mat ERK2Stepper::Evolve(vec tspan, double h, vec y) {

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
    cerr << "ERK2Stepper: Illegal h\n";
    return Y;
  }
  for (size_t tstep=0; tstep<N; tstep++) {
    if (tspan(tstep+1) < tspan(tstep)) {
      cerr << "ERK2Stepper: Illegal tspan\n";
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

      // stage 1: compute k1 at old step solution (store in k)
      if (frhs->Evaluate(t, y, k) != 0) {
        std::cerr << "ERK2Stepper::Evolve: Error in ODE RHS function\n";
        return Y;
      }

      // stage 2: compute z2 and corresponding k2 (store in k)
      z = y + (0.5*hcur)*k;
      if (frhs->Evaluate(t+0.5*hcur, z, k) != 0) {
        std::cerr << "ERK2Stepper::Evolve: Error in ODE RHS function\n";
        return Y;
      }

      // update solution
      y += (hcur*k);

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
