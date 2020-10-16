/* DIRK time stepper class implementation file.

   Class to perform time evolution of the IVP
        y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
   using an implicit Runge-Kutta time stepping method. 

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#include "dirk.hpp"

using namespace std;
using namespace arma;


//////////// DIRKResid ////////////

// DIRKResid evaluation routine
//
// Input:   z holds the current stage guesses
// Output:  resid holds the residual of each nonlinear equation
int DIRKResid::Evaluate(vec& z, vec& resid) {

  // r = z - h*a_{i,i}*f(t_{n,i},z) - yold - sum_{j=0}^{j<i} h*A_{i,j}*f(t_{n,j},zj)
  //   = z - h*a_{i,i}*f(t_{n,i},z) - a
  int ierr = frhs->Evaluate(t+c(cur_stage)*h, z, resid);    // resid = f(t_{n,i},z)
  if (ierr != 0) {
    std::cerr << "Error in ODE RHS function = " << ierr << "\n";
    return ierr;
  }
  // finish off the residual
  resid *= -(h*A(cur_stage,cur_stage));   // resid = -h*A_{i,i}*f(t_{n,i},z)
  resid += (z - a);                       // resid = z - a - h*A_{i,i}*f(t_{n,i},z)
  
  // return success
  return 0;
};



//////////// DIRKResidJac ////////////

// DIRKResidJac evaluation routine: I - h*a_{i,i}*Jrhs
//
// Input:   z holds the current stage guess
// Output:  J holds the residual Jacobian
int DIRKResidJac::Evaluate(vec& z, mat& J) {

  int ierr = Jrhs->Evaluate(t+c(cur_stage)*h, z, J);  // call Jacobian
  if (ierr != 0) {
    std::cerr << "Error in ODE RHS Jacobian function = " << ierr << "\n";
    return ierr;
  }
  J *= (-h*A(cur_stage,cur_stage));

  // add identity
  for (int i=0; i<m; i++)
    J(i,i) += 1.0;

  // return success
  return 0;
};



//////////// DIRK_Stepper ////////////


// The actual DIRK time step evolution routine
//
// Inputs:  tspan holds the current time interval, [t0, tf]
//          h holds the desired time step size
//          y holds the initial condition, y(t0)
// Outputs: the output matrix holds the computed solution at
//          all tspan values,
//            [y(t0), y(t1), ..., y(tN)]
mat DIRKStepper::Evolve(vec& tspan, double h, vec y) {

  // store sizes
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
    cerr << "DIRKStepper: Illegal h\n";
    return Y;
  }
  for (size_t tstep=0; tstep<N; tstep++) {
    if (tspan(tstep+1) < tspan(tstep)) {
      cerr << "DIRKStepper: Illegal tspan\n";
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

      // update r and rJac objects with current state information
      r.t    = t;      // copy current time into objects
      rJac.t = t;
      r.h    = hcur;   // copy current stepsize into objects
      rJac.h = hcur;
      yold = y;        // copy y into stored yold object

      // loop over stages
      for (size_t stage=0; stage<s; stage++) {

        // set initial guess for each stage solution to yold
        z = yold;

        // notify r and rJac about current stage index
        r.cur_stage = stage;
        rJac.cur_stage = stage;

        // fill in r.a with current 'old data', yold + sum_{j=0}^{i-1} h*a_{i,j}*f(t_{n,j},zj)
        r.a = yold;
        for (int j=0; j<stage; j++) {
          r.a += (hcur*A(stage,j))*K.col(j);
        }

        // call Newton method to solve for the updated solution
        int ierr = newt.Solve(z);
        if (ierr != 0) {
          std::cerr << "DIRKStepper: Error in Newton solver function = "
                    << ierr << "\n";
          return Y;
        } 

        // call IVP RHS routine to fill k (store in K)
        if (frhs->Evaluate(t+c(stage)*hcur, z, k) != 0) {
          std::cerr << "Error in ODE RHS function\n";
          return Y;
        }
        K.col(stage) = k;

      }

      // compute time-evolved solution (store in y)
      //    ynew = yold + h\sum_i (b_i * k_i)
      for (int i=0; i<s; i++) {
        y += (h*b(i)*K.col(i));
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
