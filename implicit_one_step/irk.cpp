/* IRK time stepper class implementation file.

   Class to perform time evolution of the IVP
        y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
   using an implicit Runge-Kutta time stepping method. 

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#include "irk.hpp"

using namespace std;
using namespace arma;


//////////// IRKResid ////////////

// IRKResid evaluation routine
//
// Input:   z holds the current stage guesses
// Output:  resid holds the residual of each nonlinear equation
int IRKResid::Evaluate(vec& z, vec& resid) {

  // fill in residuals for each stage as subvectors of resid

  //    first portion: zi - yold 
  for (size_t i=0; i<s; i++)
    for (size_t k=0; k<m; k++)
      resid(i*m+k) = z(i*m+k) - (*yold)(k);

  //    second portion: -h*sum[Aij*f(t+cj*h,zj)]
  for (size_t j=0; j<s; j++) {
    double tj = t + (*c)(j)*h;                // stage time
    zj = z.submat(j*m,0,(j+1)*m-1,0);         // jth stage vector
    int ierr = frhs->Evaluate(tj, zj, kj);    // call RHS function
    if (ierr != 0) {
      std::cerr << "Error in ODE RHS function = " << ierr << "\n";
      return ierr;
    }
    for (size_t i=0; i<s; i++)      // update each residual with RHS contribution
      for (size_t k=0; k<m; k++) 
        resid(i*m+k) -= h*(*A)(i,j)*kj(k);
  }

  // return success
  return 0;
};



//////////// IRKResidJac ////////////

// IRKResidJac evaluation routine
//
// Input:   z holds the current stage guesses
// Output:  J holds the overall residual Jacobian
int IRKResidJac::Evaluate(vec& z, mat& J) {

  // fill in Jacobian blocks
  J.fill(0.0);

  //    first portion: I
  for (size_t i=0; i<m*s; i++)
    J(i,i) = 1.0;

  //    second portion: -h*sum[Aij*J(t+cj*h,zj)]
  for (size_t j=0; j<s; j++) {
    double tj = t + (*c)(j)*h;              // stage time
    zj = z.submat(j*m,0,(j+1)*m-1,0);       // jth stage vector
    int ierr = Jrhs->Evaluate(tj, zj, Jj);  // call Jacobian
    if (ierr != 0) {
      std::cerr << "Error in ODE RHS Jacobian function = " << ierr << "\n";
      return ierr;
    }
    // update matrix blocks with Jacobian contribution
    for (size_t i=0; i<s; i++)
      for (size_t k=0; k<m; k++)
        for (size_t l=0; l<m; l++)
          J(i*m+l, j*m+k) -= (h * (*A)(i,j) * Jj(l,k));
  }

  // return success
  return 0;
};



//////////// IRK_Stepper ////////////

// IRK stepper solution calculation routine (given solution to nonlinear system)
//
// Inputs:  t  holds the current time
//          h  holds the current time step size
//          z  holds the stage solutions
// Output:  y  holds the resulting solution
int IRKStepper::ComputeResult(double t, double h, vec& z, vec& y) {

  // fill in result
  //    first portion: yold 
  y = yold;

  //    second portion: +h*sum[bj*f(t+cj*h,zj)]
  for (size_t j=0; j<s; j++) {
    double tj = t + c(j)*h;                 // stage time
    zj = z.submat(j*m,0,(j+1)*m-1,0);       // jth stage vector
    int ierr = frhs->Evaluate(tj, zj, kj);  // call RHS function
    if (ierr != 0) {
      std::cerr << "Error in ODE RHS function = " << ierr << "\n";
      return ierr;
    }
    y += ((h*b(j))*kj);    // update with RHS contribution
  }

  // return success
  return 0;
};


// The actual IRK time step evolution routine
//
// Inputs:  tspan holds the current time interval, [t0, tf]
//          h holds the desired time step size
//          y holds the initial condition, y(t0)
// Outputs: the output matrix holds the computed solution at
//          all tspan values,
//            [y(t0), y(t1), ..., y(tN)]
mat IRKStepper::Evolve(vec& tspan, double h, vec y) {

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
    cerr << "IRKStepper: Illegal h\n";
    return Y;
  }
  for (size_t tstep=0; tstep<N; tstep++) {
    if (tspan(tstep+1) < tspan(tstep)) {
      cerr << "IRKStepper: Illegal tspan\n";
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

      // set initial guess for each stage solution to y
      for (int j=0; j<s; j++)
        z(span(j*m,(j+1)*m-1)) = y;

      // call Newton method to solve for the updated solution
      int ierr = newt.Solve(z);
      if (ierr != 0) {
        std::cerr << "IRKStepper: Error in Newton solver function = "
                  << ierr << "\n";
        return Y;
      }

      // update current solution with IRK result
      ierr = ComputeResult(t, h, z, y);
      if (ierr != 0) {
        std::cerr << "IRKStepper: Error in ComputeResult function = " << ierr << "\n";
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
