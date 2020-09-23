/* Backward Euler time stepper class implementation file.

   Class to perform time evolution of the IVP
        y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
   using the backward Euler (implicit Euler) time stepping method.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#include "bwd_euler.hpp"

using namespace std;
using namespace arma;


// The actual backward Euler time step evolution routine
//
// Inputs:  tspan holds the current time interval, [t0, tf]
//          h holds the desired time step size
//          y holds the initial condition, y(t0)
// Outputs: the output matrix holds the computed solution at
//          all tspan values,
//            [y(t0), y(t1), ..., y(tN)]
mat BackwardEulerStepper::Evolve(vec& tspan, double h, vec y) {

  // store sizes
  size_t m = y.n_elem;
  size_t N = tspan.n_elem-1;

  // initialize output
  mat Y(m, N+1, fill::zeros);
  Y.col(0) = y;

  // reset nsteps & nnewt counters, current time value
  nsteps = 0;
  nnewt = 0;
  double t = tspan(0);

  // check for legal inputs
  if (h <= 0.0) {
    cerr << "BackwardEulerStepper: Illegal h\n";
    return Y;
  }
  for (size_t tstep=0; tstep<N; tstep++) {
    if (tspan(tstep+1) < tspan(tstep)) {
      cerr << "BackwardEulerStepper: Illegal tspan\n";
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

      // update resid and residJac objects with information on current state
      resid.t    = t;      // copy current time into objects
      residJac.t = t;
      resid.h    = hcur;   // copy current stepsize into objects
      residJac.h = hcur;
      yold = y;            // copy y into stored yold object

      // call Newton method to solve for the updated solution
      int ierr = newt.Solve(y);
      if (ierr != 0) {
        std::cerr << "BackwardEulerStepper: Error in Newton solver function = "
                  << ierr << "\n";
        return Y;
      }

      // update current time, nsteps & nnewt counters
      t += hcur;
      nsteps++;
      nnewt += newt.GetIters();

    }

    // store updated solution in output array
    Y.col(tstep+1) = y;

  }

  return Y;
}
