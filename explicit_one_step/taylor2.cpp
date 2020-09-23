/* Explicit 2nd-order Taylor series method time stepper class implementation file.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#include "taylor2.hpp"

using namespace std;
using namespace arma;

// The explicit 2nd-order Taylor method time step evolution routine
//
// Inputs:  tspan holds the time intervals, [t0, t1, ..., tN]
//          h holds the desired time step size
//          y holds the initial condition, y(t0)
// Outputs: the output matrix holds the computed solution at
//          all tspan values,
//            [y(t0), y(t1), ..., y(tN)]
mat Taylor2Stepper::Evolve(vec tspan, double h, vec y) {

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
    cerr << "Taylor2Stepper: Illegal h\n";
    return Y;
  }
  for (size_t tstep=0; tstep<N; tstep++) {
    if (tspan(tstep+1) < tspan(tstep)) {
      cerr << "Taylor2Stepper: Illegal tspan\n";
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

      // evaluate RHS function and its derivatives
      if (frhs->Evaluate(t, y, f) != 0) {
        std::cerr << "Taylor2Stepper::Evolve: Error in ODE RHS function\n";
        return Y;
      }
      if (frhs_t->Evaluate(t, y, ft) != 0) {
        std::cerr << "Taylor2Stepper::Evolve: Error in ODE f_t function\n";
        return Y;
      }
      if (frhs_y->Evaluate(t, y, fy) != 0) {
        std::cerr << "Taylor2Stepper::Evolve: Error in ODE f_y function\n";
        return Y;
      }
      fyf = fy*f;

      // update current solution,   ynew = yold + h*f + 0.5*h*h*(ft + fy*f)
      y += (hcur*f) + (0.5*hcur*hcur)*(ft + fyf);

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
