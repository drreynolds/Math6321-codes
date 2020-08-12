/* Adaptive forward Euler solver class implementation file.

   Class to perform adaptive time evolution of the IVP
        y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
   using the forward Euler (explicit Euler) time stepping method. 

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#include "adapt_euler.hpp"

using namespace arma;

inline double temporal_error(vec& f, vec& y, double rtol, double atol) {
  return(std::max(norm(f,"inf") / ( rtol * norm(y,"inf") + atol ), sqrt(eps(1.0))));
}

// Adaptive forward Euler class constructor routine
//
// Inputs:  frhs_ holds the ODE RHSFunction object, f(t,y)
//          rtol holds the desired relative solution accuracy
//          atol holds the desired absolute solution accuracy
// 
// Sets default values for adaptivity parameters, all of which may
// be modified by the user after the solver object has been created
AdaptEuler::AdaptEuler(RHSFunction& frhs_, double rtol_, 
                       double atol_, arma::vec& y) {
  frhs = &frhs_;    // set RHSFunction pointer
  rtol = rtol_;     // set tolerances
  atol = atol_;
  fn = vec(y);           // clone y to create local vectors
  y1 = vec(y);
  y2 = vec(y);
  yerr = vec(y);

  maxit = 1e6;      // set default solver parameters
  grow = 50.0;
  safe = 0.95;
  fail = 0.5;
  ONEMSM = 1.0 - sqrt(eps(1.0));
  ONEPSM = 1.0 + sqrt(eps(1.0));
  alpha = -0.5;
  fails = 0;
  steps = 0;
  error_norm = 0.0;
  h = 0.0;
};


// The adaptive forward Euler time step evolution routine
//
// Inputs:  tspan holds the current time interval, [t0, tf]
//          y holds the initial condition, y(t0)
// Outputs: y holds the computed solution, y(tf)
//
// The return value is a row vector containing all internal 
// times at which the solution was computed,
//               [t0, t1, ..., tN]
mat AdaptEuler::Evolve(vec tspan, vec y) {

  // store sizes
  size_t m = y.n_elem;
  size_t N = tspan.n_elem-1;

  // initialize output
  mat Y(m, N+1, fill::zeros);
  Y.col(0) = y;

  // reset counters, current time value
  fails = steps = 0;
  double t = tspan(0);

  // check for legal inputs
  if ((rtol < 0.0) || (atol < 0.0)) {
    std::cerr << "AdaptEuler::Evolve error: negative tolerance(s), atol = " 
	      << atol << ",  rtol = " << rtol << std::endl;
    return Y;
  }
  if ((rtol == 0.0) && (atol == 0.0)) {
    std::cerr << "AdaptEuler::Evolve error: both tolerances cannot equal zero, atol = " 
	      << atol << ",  rtol = " << rtol << std::endl;
    return Y;
  }
  for (size_t tstep=0; tstep<N; tstep++) {
    if (tspan(tstep+1) < tspan(tstep)) {
      cerr << "AdaptEuler::Evolve Illegal tspan\n";
      return Y;
    }
  }

  // get ||y'(t0)||
  if (frhs->Evaluate(t, y, fn) != 0) {
    std::cerr << "Evolve error in RHS function\n";
    return Y;
  }

  // estimate initial h value via linearization, safety factor
  error_norm = temporal_error(fn,y,rtol,atol);
  h = safe/error_norm;

  // iterate over output times
  for (size_t tstep=0; tstep<N; tstep++) {  
    
    // loop over internal steps to reach desired output time
    while ((tspan(tstep+1)-t) > sqrt(eps(tspan(tstep+1)))) {
      
      // bound internal time step
      h = std::min(h,tspan(tstep+1)-t);

      // reset both solution approximations to current solution
      y1 = y;
      y2 = y;

      // get RHS at this time, perform full/half step updates
      if (frhs->Evaluate(t, y, fn) != 0) {
        std::cerr << "Evolve error in RHS function\n";
        return Y;
      }
      y1 += h*fn;
      y2 += (0.5*h)*fn;

      // get RHS at half-step, perform half step update
      if (frhs->Evaluate(t+0.5*h, y2, fn) != 0) {
        std::cerr << "Evolve error in RHS function\n";
        return Y;
      }
      y2 += (0.5*h)*fn;

      // compute error estimate
      yerr = y2 - y1;

      // compute error estimate success factor
      error_norm = temporal_error(yerr,y2,rtol,atol);

      // if solution has too much error: reduce step size, increment failure counter, and retry
      if (error_norm > ONEPSM) {
        h *= fail;
        fails++;
        continue;
      }

      // successful step: update current time, solution, and work counter
      t += h;
      y = 2.0*y2 - y1;
      steps++;

      // pick next time step size based on this error estimate
      double eta = safe*std::pow(error_norm, alpha);    // step size estimate
      eta = std::min(eta, grow);                        // maximum growth
      h *= eta;                                         // update h

    }

    // store updated solution in output array
    Y.col(tstep+1) = y;

  }

  return Y;
}
