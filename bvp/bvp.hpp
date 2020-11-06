/* Header file to define the second-order, scalar-valued BVP:

      u'' - 2*lambda*u' + lambda^2*u = r(t),  0<t<1,

      r(t) = (4*lambda^2*exp(lambda*(1-t)))/(1+2*exp(lambda))
             + (lambda^2-pi^2)*cos(pi*t) + 2*lambda*pi*sin(pi*t)
      u(0) = (1+exp(lambda))/(1+2*exp(lambda)) + 1
      u(1) = (1+exp(lambda))/(1+2*exp(lambda)) - 1

   This problem has analytical solution

      u(t) = exp(lambda)/(1+2*exp(lambda))*(exp(lambda*(t-1))
              + exp(-lambda*t)) + cos(pi*t)

   and the stiffness may be adjusted using the real-valued
   parameter lambda<0

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#define _USE_MATH_DEFINES
#include <cmath>

// Define class for this BVP
//   Note that setup() must be called with the current value of lambda 
//   _before_ any class functions or other internal parameters are accessed.
class BVP {
public:
  double a;
  double b;
  double lambda;
  double ua;
  double ub;
  BVP(double lam) {   // constructor
    a = 0.0;
    b = 1.0;
    lambda = lam;
    ua = (1.0+exp(lambda))/(1.0+2.0*exp(lambda)) + 1.0;
    ub = (1.0+exp(lambda))/(1.0+2.0*exp(lambda)) - 1.0;
  }
  double r(double t) {
    return ((4.0*lambda*lambda*exp(lambda*(1.0-t))/(1.0+2.0*exp(lambda)))
            + (lambda*lambda - M_PI*M_PI)*cos(M_PI*t) + 2.0*lambda*M_PI*sin(M_PI*t));
  }
  double p(double t) {
    return (2.0*lambda);
  }
  double q(double t) {
    return (-lambda*lambda);
  }
  double utrue(double t) {
    return ( exp(lambda)/(1.0+2.0*exp(lambda))*(exp(lambda*(t-1.0)) + exp(-lambda*t))
             + cos(M_PI*t) );
  }
};
