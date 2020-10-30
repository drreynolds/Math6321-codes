/* Linear multistep time stepper class implementation file.

   Class to perform time evolution of the IVP
        y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
   using either an explicit or implicit linear multistep 
   time stepping method.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#include "lmm.hpp"

using namespace std;
using namespace arma;


//////////// LMM Residual ////////////

// LMMResid evaluation routine
//
// Input:   y holds the current solution guess
// Output:  resid holds the implicit residual
int LMMResid::Evaluate(vec& y, vec& resid) {

  // evaluate RHS function at new time (store in resid)
  if (abs(b(0)) > sqrt(eps(1.0))) {
    int ierr = frhs->Evaluate(t+h, y, resid);
    if (ierr != 0) {
      std::cerr << "LMMResid: Error in ODE RHS function = " << ierr << "\n";
      return ierr;
    }
    resid *= (-h*b(0));         // resid = -h*b_0*f(t+h,y)
  } else {
    resid.fill(0.0);
  }

  // combine pieces to fill residual, sum[a_j y_{n-j}] - h*sum[b_j*f_{n-j}]
  resid += (a(0)*y);              // resid = a_0*y - h*b_0*f(t+h,y)
  for (int j=1; j<a.n_elem; j++)  // (add remaining terms in sum)
    resid += ( a(j) * yold->col(j-1) - (h*b(j)) * fold->col(j-1) );

  // if explicit:  ynew = 1/a(0)*[ -sum[a_j y_{n-j}] + h*sum[b_j*f_{n-j}] ]
  //                    = -1/a(0)*resid

  // return success
  return 0;
}


//////////// LMM Residual Jacobian ////////////

// Jacobian evaluation routine
int LMMResidJac::Evaluate(vec& y, mat& J) {

  // evaluate RHS function Jacobian (store in J)
  int ierr = Jrhs->Evaluate(t+h, y, J);
  if (ierr != 0) {
    std::cerr << "Error in ODE RHS Jacobian function = " << ierr << "\n";
    return ierr;
  }
  // combine pieces to fill residual Jacobian
  J *= (-b0*h);                     // J = -b0*h*Jrhs
  for (int i=0; i<J.n_rows; i++)    // J = I - beta*h*Jrhs
    J(i,i) += 1.0;

  // return success
  return 0;
}

//////////// LMM Time Stepper ////////////


// utility routine to set up internal multistep structures
int LMMStepper::Initialize(double t0, double h, mat& y0) {

  // ensure that supplied y0 is compatible with LMM
  if (y0.n_rows != yold.n_rows) {
    std::cerr << "LMMStepper::Initialize error -- incompatible y0 (num rows)\n";
    return -1;
  }
  if (y0.n_cols != yold.n_cols) {
    std::cerr << "LMMStepper::Initialize error -- incompatible y0 (num cols)\n";
    return -1;
  }

  // fill fold matrix
  for (int j=0; j<yold.n_cols; j++) {
    yold.col(j) = y0.col(j);
    r.yn = y0.col(j);
    int ierr = frhs->Evaluate(t0-j*h, r.yn, r.fn);
    if (ierr != 0) {
      std::cerr << "LMMStepper::Initialize error -- ODE RHS Evaluate, ierr = " << ierr << "\n";
      return ierr;
    }
    fold.col(j) = r.fn;
  }

  // return success
  return 0;
}

// utility routine to handle updates of "old" solutions and right-hand sides
//
// Inputs:  tnew  the current time for the new solution
//          ynew  the new solution
int LMMStepper::UpdateHistory(double tnew, vec& ynew) {

  // update columns of yold and fold, starting at oldest and moving to newest
  for (int icol=fold.n_cols-1; icol>0; icol--) {
    fold.col(icol) = fold.col(icol-1);
    yold.col(icol) = yold.col(icol-1);
  }

  // fill first column of yold with ynew
  yold.col(0) = ynew;

  // evaluate RHS function at new time (store in first column of fold matrix)
  int ierr = frhs->Evaluate(tnew, ynew, r.fn);
  if (ierr != 0) {
    std::cerr << "LMMStepper::UpdateHistory error in ODE RHS function = " << ierr << "\n";
    return ierr;
  }
  fold.col(0) = r.fn;

  return 0;
};


// The actual LMM time step evolution routine
//
// Inputs:  tspan holds the current time interval, [t0, tf]
//          h holds the desired time step size
//          y holds the set of initial conditions, [y_0, y_{-1}, ..., y_{-k+1}]
// Outputs: the output matrix holds the computed solution at
//          all tspan values,
//            [y(t0), y(t1), ..., y(tN)]
mat LMMStepper::Evolve(vec& tspan, double h, mat& y0) {

  // store sizes
  size_t m = y0.n_rows;
  size_t N = tspan.n_elem-1;

  // initialize output
  mat Y(m, N+1, fill::zeros);
  Y.col(0) = y0.col(0);

  // reset nsteps & nnewt counters, current time value
  nsteps = 0;
  nnewt = 0;
  double t = tspan(0);

  // set flag indicating whether LMM is implicit
  bool implicit = (abs(b(0)) > sqrt(eps(1.0)));

  // check for legal inputs
  if (h <= 0.0) {
    cerr << "LMMStepper::Evolve -- Illegal h\n";
    return Y;
  }
  for (size_t tstep=0; tstep<N; tstep++) {
    if (tspan(tstep+1) < tspan(tstep)) {
      cerr << "LMMStepper::Evolve -- Illegal tspan\n";
      return Y;
    }
  }

  // initialize internal LMM structures
  if (Initialize(t, h, y0) != 0) {
    std::cerr << "LMMStepper::Evolve -- error initializing LMM stepper\n";
    return Y;
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

      // update r with current information
      r.Update(yold, fold, t, hcur);

      ////// implicit LMM //////
      if (implicit) {

        // update rJac with current state information
        rJac.Update(t, hcur);

        // update Newton with current residual function
        newt.UpdatePointers(r, rJac, w);

        // set initial guess for this step
        yn = yold.col(0);

        // call Newton method to solve for the updated solution
        if (newt.Solve(yn) != 0) {
          std::cerr << "LMMStepper::Evolve -- Newton solver error\n";
          return Y;
        }

        // update current time, nsteps & nnewt counters
        t += hcur;
        nsteps++;
        nnewt += newt.GetIters();

      ////// explicit LMM //////
      } else {

        // evaluate residual with 'guess' set to zero
        yn.fill(0.0);
        if (r.Evaluate(yn, r.fn) != 0) {
          std::cerr << "LMMStepper::Evolve -- Error in ODE RHS function\n";
          return Y;
        }

        // set updated solution to equal -fn/(a(0))
        yn = (-1.0/a(0))*(r.fn);

        // update current time, nsteps counter
        t += hcur;
        nsteps++;

      }

      // update LMM structures
      if (UpdateHistory(t, yn) != 0) {
        std::cerr << "LMMStepper::Evolve -- error updating LMM structures\n";
        return Y;
      }

    }

    // store updated solution in output array
    Y.col(tstep+1) = yn;

  }

  return Y;

}
