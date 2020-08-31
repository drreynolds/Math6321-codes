/* Forward Euler time stepper class implementation file.

   Class to perform time evolution of the IVP
        y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
   using the forward Euler (explicit Euler) time stepping method.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020 */

#include "fwd_euler.hpp"

using namespace std;
using namespace arma;

// The forward Euler time step evolution routine
//
// Inputs:  tspan holds the time intervals, [t0, t1, ..., tN]
//          h holds the desired time step size
//          y holds the initial condition, y(t0)
// Outputs: the output matrix holds the computed solution at
//          all tspan values,
//            [y(t0), y(t1), ..., y(tN)]
mat ForwardEulerStepper::Evolve(vec tspan, double h, vec y) {

  // store sizes
  size_t m = y.n_elem;
  size_t N = tspan.n_elem-1;

  // initialize output
  mat Y(m, N+1, fill::zeros);
  Y.col(0) = y;

  // reset nsteps counter, current time value
  nsteps = 0;
  double t = tspan(0);

  // set floating-point roundoff parameter
  double ONEMSM = 1.0 - sqrt(eps(1.0));

  // check for legal inputs
  if (h <= 0.0) {
    cerr << "ForwardEulerStepper: Illegal h\n";
    return Y;
  }
  for (size_t tstep=0; tstep<N; tstep++) {
    if (tspan(tstep+1) < tspan(tstep)) {
      cerr << "ForwardEulerStepper: Illegal tspan\n";
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

      // compute ODE RHS
      if (frhs->Evaluate(t, y, f) != 0) {
        cerr << "ForwardEulerStepper: Error in ODE RHS function\n";
        return Y;
      }

      // update solution with forward Euler step
      y += (hcur*f);

      // update current time
      t += hcur;

    }

    // store updated solution in output array
    Y.col(tstep+1) = y;

  }

  return Y;
}
