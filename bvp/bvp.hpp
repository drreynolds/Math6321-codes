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

#include <iostream>
#include <iomanip>

#define PI 3.141592653589793


// Define class for this BVP:
// defines f(t,u,u'), f_u(t,u,u') and f_u'(t,u,u') functions,
// a function for the true solution, as well as auxiliary
// functions for use inside of these routines.
class BVP {
public:
  double a;
  double b;
  double lambda;
  double ua;
  double ub;
  void setup(double lam) {
    a = 0.0;
    b = 1.0;
    lambda = lam;
    ua = (1.0+exp(lambda))/(1.0+2.0*exp(lambda)) + 1.0;
    ub = (1.0+exp(lambda))/(1.0+2.0*exp(lambda)) - 1.0;
  }
  double r(double t) {
    return ((4.0*lambda*lambda*exp(lambda*(1.0-t))/(1.0+2.0*exp(lambda)))
            + (lambda*lambda - PI*PI)*cos(PI*t) + 2.0*lambda*PI*sin(PI*t));
  }
  double p(double t) {
    return (2.0*lambda);
  }
  double q(double t) {
    return (-lambda*lambda);
  }
  double utrue(double t) {
    return ( exp(lambda)/(1.0+2.0*exp(lambda))*(exp(lambda*(t-1.0)) + exp(-lambda*t))
             + cos(PI*t) );
  }
};
