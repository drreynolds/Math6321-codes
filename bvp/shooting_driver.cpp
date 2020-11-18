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
       involves solution of an augmented first-order IVP system; for
       that we use our adaptive RKF solver.

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


// Define a small 'data' class for the shooting method
//   Since both the residual and its Jacobian require evolution of an IVP
//   (the Jacobian IVP RHS is the linearized version of that used for the
//   residual), and since Newton's method always evaluates the residual
//   _before_ the Jacobian, within the residual calculation we evolve one
//   augmented system for both the shooting method residual and the two
//   augmented IVPs for the Jacobian.  The Jacobian-specific data is then
//   stored in this class for reuse by the Jacbian evaluation routine.
class ShootingData {
public:
  vec Y0;
  vec Y1;
};

// Define IVP right-hand side function class (evaluates the IVPs for
// both the residual and Jacobian)
class IVPRhs: public RHSFunction {
  BVP *bvp;               // BVP object
public:
  IVPRhs(BVP& bvp_) {      // constructor
    bvp = &bvp_;
  }
  int Evaluate(double t, vec& y, vec& f) {
    f(0) = y(1);
    f(1) = bvp->p(t)*y(1) + bvp->q(t)*y(0) + bvp->r(t);
    f(2) = y(3);
    f(3) = bvp->p(t)*y(3) + bvp->q(t)*y(2);
    f(4) = y(5);
    f(5) = bvp->p(t)*y(5) + bvp->q(t)*y(4);
    return 0;
  }
};

// Define nonlinear residual and Jacobian classes:
//    residual function class -- instantiates a ResidualFunction
class ShootingResid: public ResidualFunction {
public:
  BVP *bvp;                    // BVP object
  ShootingData *sdata;         // ShootingData object
  AdaptRKF *rkf;               // Runge-Kutta-Fehlberg IVP solver
  ShootingResid(AdaptRKF& rkf_, BVP& bvp_, ShootingData& sdata_) {   // constructor
    rkf = &rkf_;  bvp = &bvp_;  sdata = &sdata_;
  }
  int Evaluate(vec& c, vec& r) {
    // evolve the augmented IVP
    vec y(6);
    y(span(0,1)) = c;
    y(2) = 1.0;
    y(3) = 0.0;
    y(4) = 0.0;
    y(5) = 1.0;
    vec tspan(2);
    tspan(0) = bvp->a;
    tspan(1) = bvp->b;
    mat Y = rkf->Evolve(tspan, y);

    // evaluate the nonlinear residual
    r(0) = c(0) - bvp->ua;     // left boundary condition
    r(1) = Y(0,1) - bvp->ub;   // right boundary condition

    // store Jacobian-related results in sdata object
    sdata->Y0 = Y(span(2,3),1);
    sdata->Y1 = Y(span(4,5),1);
    return 0;
  }
};

//    residual Jacobian class -- instantiates a ResidualJacobian
class ShootingJac: public ResidualJacobian {
public:
  ShootingData *sdata;         // ShootingData object
  ShootingJac(ShootingData& sdata_) {   // constructor
    sdata = &sdata_;
  }
  int Evaluate(vec& c, mat& J) {
    J(0,0) = 1.0;
    J(0,1) = 0.0;
    J(1,0) = sdata->Y0(0);
    J(1,1) = sdata->Y1(0);
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

  // define BVP and ShootingData objects
  BVP bvp(lambda);
  ShootingData sdata;

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
    vec rkf_atol(6);  rkf_atol.fill(rkf_rtol(i)/1000);

    // output problem information
    cout << "\nShooting method for BVP with lambda = " << lambda << ":\n"
         << "  newt_tol = " << newt_tol
         << ",  rkf_rtol = " << rkf_rtol(i)
         << ",  rkf_atol  = " << rkf_atol(0) << "\n";

    // create IVP solvers, residual and Jacobian objects
    IVPRhs ivp_rhs(bvp);
    vec y(6);    // empty vector of appropriate size
    AdaptRKF ivp_rkf(ivp_rhs, rkf_rtol(i), rkf_atol, y);
    ShootingResid resid(ivp_rkf, bvp, sdata);
    ShootingJac jac(sdata);

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
    y.fill(0.0);
    y(span(0,1)) = c;
    mat Y = ivp_rkf.Evolve(tspan, y);

    // output maximum error
    vec uerr = abs(trans(Y.row(0))-utrue);
    cout << "  Maximum BVP solution error = " << std::setprecision(4)
       << uerr.max() << "\n";
  }

  return 0;
}
