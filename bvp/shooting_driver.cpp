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
  IVPRhs(BVPRhs bvp_) {   // constructor
    bvp = &bvp_;
  }
  int Evaluate(double t, vec& y, vec& f) {
    f[0] = y[1];                                 // u'
    f[1] = bvp->f_Eval(t, y[0], y[1]);           // u''
    f[2] = y[3];                                 // eta'
    f[3] = bvp->f_u_Eval(t, y[0], y[1])*y[2]     // eta''
         + bvp->f_du_Eval(t, y[0], y[1])*y[3];
    return 0;
  }
};


// Define IVP right-hand side function class:
// evolves y = {u, u', eta, eta'}
class IVPRhs: public RHSFunction {
  BVP *bvp;               // BVP object
public:
  IVPRhs(BVPRhs bvp_) {   // constructor
    bvp = &bvp_;
  }
  int Evaluate(double t, vector<double>& y, vector<double>& f) {
    f[0] = y[1];                                 // u'
    f[1] = bvp->f_Eval(t, y[0], y[1]);           // u''
    f[2] = y[3];                                 // eta'
    f[3] = bvp->f_u_Eval(t, y[0], y[1])*y[2]     // eta''
         + bvp->f_du_Eval(t, y[0], y[1])*y[3];
    return 0;
  }
};


// Define nonlinear residual and Jacobian classes:
//    residual function class -- instantiates a ResidualFunction
class ShootingResid: public ResidualFunction {
public:
  AdaptRKF *RKF;               // Runge-Kutta-Fehlberg IVP solver
  vector<double> tspan;        // BVP solution interval
  vector<double> a, b, c, g;   // Boundary condition parameters
  ShootingResid(AdaptRKF& RKF_,
                vector<double> tspan_,
                vector<double> a_,
                vector<double> b_,
                vector<double> c_,
                vector<double> g_) {   // constructor
    RKF = &RKF_;  tspan = tspan_;
    a = a_;  b = b_;  c = c_;  g = g_;
  }
  int Evaluate(vector<double>& s,    // evaluate the residual, r(s)
               vector<double>& r) {
    // evolve the IVP
    vector<double> y = {a[1]*s[0] - c[1]*g[0],
                        a[0]*s[0] - c[0]*g[0],
                        a[1], a[0]};
    vector<double> tvals = RKF->Evolve(tspan, y);
    // evaluate the nonlinear residual
    r[0] = b[0]*y[0] + b[1]*y[1] - g[1];
    return 0;
  }
};

//    residual Jacobian class -- instantiates a ResidualJacobian
class ShootingJac: public ResidualJacobian {
public:
  AdaptRKF *RKF;                // Runge-Kutta-Fehlberg IVP solver
  vector<double> tspan;         // BVP solution interval
  vector<double> a, b, c, g;    // Boundary condition parameters
  ShootingJac(AdaptRKF& RKF_,
              vector<double> tspan_,
              vector<double> a_,
              vector<double> b_,
              vector<double> c_,
              vector<double> g_) {   // constructor
    RKF = &RKF_;  tspan = tspan_;
    a = a_;  b = b_;  c = c_;  g = g_;
  }
  int Evaluate(vector<double>& s, Matrix& J) {
    J = 0.0;
    // evolve the IVP
    vector<double> y = {a[1]*s[0] - c[1]*g[0],
                        a[0]*s[0] - c[0]*g[0],
                        a[1], a[0]};
    vector<double> tvals = RKF->Evolve(tspan, y);
    // evaluate the Jacobian
    J(0,0) = b[0]*y[2] + b[1]*y[3];
    return 0;
  }
};



// main routine
int main(int argc, char **argv) {

  // define BVP
  BVPRhs bvp;
  vector<double> tspan = {0.0, 2.0*M_PI};
  vector<double> bc_a = {2.0, -1.0};
  vector<double> bc_b = {1.0, -1.0};
  vector<double> bc_g = {-2.0, 5.0};
  vector<double> bc_c(2);
  if (bc_a[0] != 0.0) {
    bc_c = {1.0, (bc_a[1] - 1.0)/bc_a[0]};
  } else {
    bc_c = {(bc_a[0] - 1.0)/bc_a[1], 1.0};
  }

  // set shooting method tolerances from command line
  double newt_rtol = 1.0e-3;
  if (argc > 1)
    newt_rtol = atof(argv[1]);
  double newt_atol = newt_rtol / 100;
  double rkf_rtol = newt_rtol / 20;
  double rkf_atol = newt_atol / 20;

  std::cout << "\nShooting method for BVP:\n";
  std::cout << "  newt_rtol = " << newt_rtol << "\n";
  std::cout << "  newt_atol = " << newt_atol << "\n";
  std::cout << "  rkf_rtol  = " << rkf_rtol << "\n";
  std::cout << "  rkf_atol  = " << rkf_atol << "\n";


  // create IVP, IVP solver, residual and Jacobian objects
  IVPRhs ivp(bvp);
  vector<double> y(4);    // empty vector of appropriate size
  AdaptRKF rkf(ivp, rkf_rtol, rkf_atol, y);
  ShootingResid resid(rkf, tspan, bc_a, bc_b, bc_c, bc_g);
  ShootingJac jac(rkf, tspan, bc_a, bc_b, bc_c, bc_g);

  // initial guess
  vector<double> s = {0.0};

  // create Newton solver object, call solver, output solution
  NewtonSolver newt(resid, jac, newt_rtol, newt_atol, 20, s, true);
  std::cout << "\nCalling Newton solver to solve shooting method:\n";
  if (newt.Solve(s) != 0) {
    std::cerr << "Newton failure\n";
    return 1;
  }

  // output final 's' value, re-run IVP to generate BVP solution
  std::cout << "\nNewton solution: " << std::setprecision(16)
            << s[0] << "\n";

  y = {bc_a[1]*s[0] - bc_c[1]*bc_g[0],
       bc_a[0]*s[0] - bc_c[0]*bc_g[0],
       bc_a[1], bc_a[0]};                     // IVP initial condition
  double tcur = tspan[0];                     // IVP time counter
  int Nt = 201;                               // solution storage
  double dtout = (tspan[1]-tspan[0])/(Nt-1);  // IVP output frequency

  // loop over output steps: call solver and check solution
  double maxerr = 0.0;
  while (tcur < 0.99999*tspan[1]) {
    vector<double> thisspan = {tcur, std::min( tcur + dtout, tspan[1])};
    vector<double> tvals = rkf.Evolve(thisspan, y);
    tcur = tvals.back();
    maxerr = std::max(maxerr, fabs(bvp.utrue(tcur) - y[0]));
  }

  // output maximum error and return
  cout << "maximum BVP solution error = " << std::setprecision(4)
       << maxerr << "\n";
  cout << "total RKF steps, failures = " << rkf.steps << ", "
       << rkf.fails << "\n\n";

  return 0;
}
