/* Main routine to run a shooting method for solution of a
   second-order, scalar-valued BVP:

      u'' = p(t)*u' + q(t)*u + r(t),  a<t<b,
      u(a) = ua,  u(b) = ub

   where the problem has stiffness that may be adjusted using
   the real-valued parameter lambda<0 [read from the command line]


   This driver attempts to solve the problem using a single
   shooting method:

   (a) convert BVP to first-order IVP system
   (b) use Newton's method to solve for the shooting parameter u'(0)=s
   (c) within Newton's method, each residual/Jacobian evaluation
       involves solution of the first-order IVP system; for that
       we use our adaptive RKF solver.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#include <iostream>
#include <iomanip>
#include "bvp.hpp"
#include "rhs.hpp"
#include "newton.hpp"
#include "adapt_rkf.hpp"

using namespace std;
using namespace arma;


// Define main IVP right-hand side function class
class IVPRhs: public RHSFunction {
  BVP *bvp;               // BVP object
public:
  IVPRhs(BVP& bvp_) {      // constructor
    bvp = &bvp_;
  }
  int Evaluate(double t, vec& y, vec& f) {
    f(0) = y(1);
    f(1) = bvp->p(t)*y(1) + bvp->q(t)*y(0) + bvp->r(t);
    return 0;
  }
};

// Define auxiliary IVP right-hand side function class
class IVPJacRhs: public RHSFunction {
  BVP *bvp;               // BVP object
public:
  IVPJacRhs(BVP& bvp_) {   // constructor
    bvp = &bvp_;
  }
  int Evaluate(double t, vec& y, vec& f) {
    f(0) = y(1);
    f(1) = bvp->p(t)*y(1) + bvp->q(t)*y(0);
    return 0;
  }
};


// Define nonlinear residual and Jacobian classes:
//    residual function class -- instantiates a ResidualFunction
class ShootingResid: public ResidualFunction {
public:
  BVP *bvp;                    // BVP object
  AdaptRKF *rkf;               // Runge-Kutta-Fehlberg IVP solver
  ShootingResid(AdaptRKF& rkf_, BVP& bvp_) {   // constructor
    rkf = &rkf_;  bvp = &bvp_;
  }
  int Evaluate(vec& c, vec& r) {
    // evolve the IVP
    vec y = c;
    vec tspan(2);
    tspan(0) = bvp->a;
    tspan(1) = bvp->b;
    mat Y = rkf->Evolve(tspan, y);
    // evaluate the nonlinear residual
    r(0) = c(0) - bvp->ua;     // left boundary condition
    r(1) = Y(0,1) - bvp->ub;   // right boundary condition
    return 0;
  }
};

//    residual Jacobian class -- instantiates a ResidualJacobian
class ShootingJac: public ResidualJacobian {
public:
  BVP *bvp;                     // BVP object
  AdaptRKF *rkf;                // Runge-Kutta-Fehlberg IVP solver
  mat Jsaved;                   // stored Jacobian
  ShootingJac(AdaptRKF& rkf_, BVP& bvp_) {   // constructor
    rkf = &rkf_;  bvp = &bvp_;

    // note that since the problem is linear in y, the residual 
    // Jacobian is fixed.  We thus perform all Jacobian-related
    // solves now, and store the results for later.
    Jsaved = mat(2,2);
    Jsaved(0,0) = 1.0;
    Jsaved(0,1) = 0.0;
    vec tspan(2);
    tspan(0) = bvp->a;
    tspan(1) = bvp->b;
    vec y(2);
    y(0) = 1.0;  y(1) = 0.0;
    mat Y = rkf->Evolve(tspan, y);
    Jsaved(1,0) = Y(0,1);
    y(0) = 0.0;  y(1) = 1.0;
    Y = rkf->Evolve(tspan, y);
    Jsaved(1,1) = Y(0,1);
  }
  int Evaluate(vec& c, mat& J) {
    J = Jsaved;
    return 0;
  }
};


// main routine
int main(int argc, char **argv) {

  // get lambda from the command line, otherwise set to -10
  double lambda;
  if (argc > 1)
    lambda = atof(argv[1]);
  if (lambda >= 0.0)
    lambda = -10.0;

  // set final solution resolution
  int N = 1001;

  // define BVP object
  BVP bvp(lambda);
  
  // compute/store analytical solution
  vec tspan = linspace(bvp.a, bvp.b, N);
  vec utrue(N);
  for (int i=0; i<N; i++)
    utrue(i) = bvp.utrue(tspan(i));

  // since h(c) is linear, then there's no point in running the 
  // shooting method for various Newton tolerances, so just use one
  double newt_tol = 1.0e-3;

  // loop over various inner IVP tolerances
  vec rkf_rtol("1.e-5, 1.e-8, 1.e-11");
  for (int i=0; i<rkf_rtol.n_elem; i++) {

    // set tight IVP tolerances
    vec rkf_atol(2);  rkf_atol.fill(rkf_rtol(i)/1000);

    // output problem information
    cout << "\nShooting method for BVP with lambda = " << lambda << ":\n"
         << "  newt_tol = " << newt_tol
         << ",  rkf_rtol = " << rkf_rtol(i)
         << ",  rkf_atol  = " << rkf_atol(0) << "\n";

    // create IVP solvers, residual and Jacobian objects
    IVPRhs    ivp_rhs(bvp);
    IVPJacRhs jac_ivp_rhs(bvp);
    vec y(2);    // empty vector of appropriate size
    AdaptRKF ivp_rkf(ivp_rhs, rkf_rtol(i), rkf_atol, y);
    AdaptRKF jac_ivp_rkf(jac_ivp_rhs, rkf_rtol(i), rkf_atol, y);
    ShootingResid resid(ivp_rkf, bvp);
    ShootingJac jac(jac_ivp_rkf, bvp);

    // initial guess
    vec c(2);
    c(0) = bvp.ua;
    c(1) = 0.0;

    // create Newton solver object, call solver, output solution
    vec w(2);  w.fill(1.0);   // set trivial error weight vector
    NewtonSolver newt(resid, jac, newt_tol, w, 20, c, true);
    cout << "  Calling Newton solver:\n";
    if (newt.Solve(c) != 0)
      cerr << "  Warning: Newton convergence failure\n";

    // output final c value, re-run IVP to generate BVP solution
    cout << "  Newton solution: " << setprecision(16) << trans(c);
    mat Y = ivp_rkf.Evolve(tspan, c);

    // output maximum error
    vec uerr = abs(trans(Y.row(0))-utrue);
    cout << "  Maximum BVP solution error = " << std::setprecision(4)
       << uerr.max() << "\n";
  }
  
  return 0;
}
