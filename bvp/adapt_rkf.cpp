/* Adaptive Runge-Kutta-Fehlberg solver class implementation file.

   Class to perform time evolution of the IVP
        y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
   using the Runge-Kutta-Fehlberg time stepping method.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#include "adapt_rkf.hpp"

using namespace arma;

// Adaptive Runge-Kutta-Fehlberg class constructor routine
//
// Inputs:  frhs_ holds the ODE RHSFunction object, f(t,y)
//          rtol holds the desired relative solution accuracy
//          atol holds the desired absolute solution accuracy
AdaptRKF::AdaptRKF(RHSFunction& frhs_, double rtol_,
                   arma::vec& atol_, arma::vec& y) {
  frhs = &frhs_;    // set RHSFunction pointer
  atol = &atol_;    // set absolute tolerance pointer
  rtol = rtol_;     // copy relative tolerance
  z  = vec(y);      // clone y to create local vectors
  w  = vec(y);
  yt = vec(y);
  k  = vec(y);
  K  = mat(y.n_elem, 6);

  A = mat(6,6);    // RKF Butcher table
  A(1,0) = 1.0/4.0;
  A(2,0) = 3.0/32.0;
  A(2,1) = 9.0/32.0;
  A(3,0) = 1932.0/2197.0;
  A(3,1) = -7200.0/2197.0;
  A(3,2) = 7296.0/2197.0;
  A(4,0) = 439.0/216.0;
  A(4,1) = -8.0;
  A(4,2) = 3680.0/513.0;
  A(4,3) = -845.0/4104.0;
  A(5,0) = -8.0/27.0;
  A(5,1) = 2.0;
  A(5,2) = -3544.0/2565.0;
  A(5,3) = 1859.0/4104.0;
  A(5,4) = -11.0/40.0;
  b = vec(6);
  b(0) = 16.0/135.0;
  b(1) = 0.0;
  b(2) = 6656.0/12825.0;
  b(3) = 28561.0/56430.0;
  b(4) = -9.0/50.0;
  b(5) = 2.0/55.0;
  b2 = vec(6);
  b2(0) = 25.0/216.0;
  b2(1) = 0.0;
  b2(2) = 1408.0/2565.0;
  b2(3) = 2197.0/4104.0;
  b2(4) = -1.0/5.0;
  b2(5) = 0.0;
  c = vec(6);
  c(0) = 0.0;
  c(1) = 1.0/4.0;
  c(2) = 3.0/8.0;
  c(3) = 12.0/13.0;
  c(4) = 1.0;
  c(5) = 1.0/2.0;

  maxit = 1e6;               // set default solver parameters
  bias = 2.0;
  grow = 50.0;
  safe = 0.95;
  ONEPSM = 1.0 + sqrt(eps(1.0));
  p = 4;
  fails = 0;
  steps = 0;
  error_norm = 0.0;
  h = 0.0;
};

// Error weight vector utility routine
void AdaptRKF::error_weight(vec& y, vec& w) {
  for (size_t i=0; i<y.size(); i++)
    w(i) = bias / ((*atol)(i) + rtol * std::abs(y(i)));
}

// Single step of Runge-Kutta-Fehlberg method (with error estimate)
//
// Inputs:  t holds the current time
//          h holds the current time step size
//          y holds the current solution
// Outputs: y holds the updated solution
//          error_norm (stored in class) is  ||y1(t+h) - y2(t+h)||
//
// The return value is an integer indicating success/failure,
// with 0 indicating success, and nonzero failure.
int AdaptRKF::Step(double t, double h, arma::vec& y) {

  // construct RK stage solutions, and evaluate stage RHS vectors
  for (int i=0; i<c.size(); i++) {
    z = y;
    for (int j=0; j<i; j++)
      z += (h*A(i,j))*K.col(j);
    if (frhs->Evaluate(t+c(i)*h, z, k) != 0) {
      std::cerr << "Step: Error in ODE RHS function\n";
      return 1;
    }
    K.col(i) = k;
  }

  // update solution
  for (int j=0; j<c.size(); j++)
    y += (h*b(j))*K.col(j);

  // compute error estimate (and norm)
  z.fill(0.0);
  for (int j=0; j<c.size(); j++)
    z += (h*(b(j)-b2(j)))*K.col(j);
  error_norm = norm(z%w,"inf");
  error_norm = std::max(error_norm, 1.e-8);

  // return success
  return 0;

}

// The adaptive Runge-Kutta-Fehlberg time step evolution routine
//
// Inputs:  tspan holds the current time interval, [t0, tf]
//          y holds the initial condition, y(t0)
// Outputs: the output matrix holds the computed solution at
//          all tspan values,
//            [y(t0), y(t1), ..., y(tN)]
mat AdaptRKF::Evolve(vec tspan, vec y) {

  // store sizes
  size_t m = y.n_elem;
  size_t N = tspan.n_elem-1;

  // initialize output
  mat Y(m, N+1, fill::zeros);
  Y.col(0) = y;

  // reset counters, current time value
  fails = steps = 0;
  double t = tspan(0);

  // check for legal time span
  for (size_t tstep=0; tstep<N; tstep++) {
    if (tspan(tstep+1) < tspan(tstep)) {
      cerr << "AdaptEuler::Evolve Illegal tspan\n";
      return Y;
    }
  }

  // initialize error weight vector, and check for legal tolerances
  error_weight(y,w);

  // get ||y'(t0)||
  if (frhs->Evaluate(t, y, k) != 0) {
    std::cerr << "Evolve error in RHS function\n";
    return Y;
  }

  // estimate initial h value via linearization, safety factor
  error_norm = norm(k%w,"inf");
  error_norm = std::max(error_norm, 1.e-8);
  h = safe/error_norm;

  // iterate over output times
  for (size_t tstep=0; tstep<N; tstep++) {

    // loop over internal steps to reach desired output time
    while ((tspan(tstep+1)-t) > sqrt(eps(tspan(tstep+1)))) {

      // enforce maxit
      if (steps+fails > maxit) {
        std::cerr << "Exceeded maximum allowed iterations\n";
        return Y;
      }

      // bound internal time step
      h = std::min(h,tspan(tstep+1)-t);

      // reset temporary solution to current solution, and take RKF step
      yt = y;
      if (Step(t, h, yt) != 0) {
        std::cerr << "Evolve error in Step() function\n";
        return Y;
      }

      // check error
      if (error_norm < ONEPSM) {  // successful step

        // update current time, solution, error weights, and work counter
        t += h;
        y = yt;
        error_weight(y,w);
        steps++;

      } else {                    // failed step
        fails++;
      }

      // pick next time step size based on this error estimate
      double eta = safe*std::pow(error_norm, -1.0/(p+1));   // step size growth factor
      eta = std::min(eta, grow);                            // limit maximum growth
      h *= eta;                                             // update h

    }

    // store updated solution in output array
    Y.col(tstep+1) = y;

  }

  return Y;
}
