/* Main routine to test the higher-order one-step methods

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020 */

#include <iostream>
#include "heun.hpp"
#include "erk2.hpp"
#include "erk3.hpp"
#include "erk4.hpp"
#include "taylor2.hpp"

using namespace std;
using namespace arma;

class RHS: public RHSFunction {
public:
  int Evaluate(double t, vec& y, vec& f) {    // evaluates the RHS function, f(t,y)
    f = -y*exp(-t);
    return 0;
  }
};

class RHSt: public RHSFunction {
public:
  int Evaluate(double t, vec& y, vec& ft) {    // evaluates the RHS function, f_t(t,y)
    ft = y*exp(-t);
    return 0;
  }
};

class RHSy: public RHSJacobian {
public:
  int Evaluate(double t, vec& y, mat& fy) {    // evaluates the RHS Jacobian, f_y(t,y)
    fy(0) = -exp(-t);
    return 0;
  }
};

// Convenience function for analytical solution
vec ytrue(const double t) {
  vec yt(1);
  yt(0) = exp(exp(-t)-1.0);
  return yt;
};


// main routine
int main() {

  // time steps to try
  vec h("0.5 0.1 0.05 0.01 0.005 0.001 0.0005");

  // vector for errors at each h
  vec e(h.n_elem);

  // set problem information
  vec y0("1.0");
  double t0 = 0.0;
  double Tf = 1.0;

  // set desired output times
  int Nout = 3;  // includes initial condition
  vec tspan = linspace(t0, Tf, Nout);

  // create ODE RHS function objects
  RHS  f;
  RHSt ft;
  RHSy fy;

  // create true solution results
  mat Ytrue(1,Nout);
  for (size_t i=0; i<Nout; i++)
    Ytrue.col(i) = ytrue(tspan(i));
  
  //---- Taylor 2 ----
  cout << "\nTaylor2:\n";
  Taylor2Stepper T2(f, ft, fy, y0);
  for (size_t ih=0; ih<h.n_elem; ih++) {

    // call stepper
    cout << "  h = " << h(ih) << ":";
    mat Y = T2.Evolve(tspan, h(ih), y0);

    // output solution, errors, and overall error
    mat Yerr = abs(Y-Ytrue);
    e(ih) = Yerr.max();
    if (ih > 0) {
      cout << "  Max error = " << e(ih) << ",  conv rate = " 
           << log(e(ih)/e(ih-1))/log(h(ih)/h(ih-1)) << endl;
    } else {
      cout << "  Max error = " << e(ih) << endl;
    }

  }


  //---- Heun ----
  cout << "\nHeun:\n";
  HeunStepper H(f, y0);
  for (size_t ih=0; ih<h.n_elem; ih++) {

    // call stepper
    cout << "  h = " << h(ih) << ":";
    mat Y = H.Evolve(tspan, h(ih), y0);

    // output solution, errors, and overall error
    mat Yerr = abs(Y-Ytrue);
    e(ih) = Yerr.max();
    if (ih > 0) {
      cout << "  Max error = " << e(ih) << ",  conv rate = " 
           << log(e(ih)/e(ih-1))/log(h(ih)/h(ih-1)) << endl;
    } else {
      cout << "  Max error = " << e(ih) << endl;
    }

  }


  //---- ERK 2 ----
  cout << "\nERK2:\n";
  ERK2Stepper ERK2(f, y0);
  for (size_t ih=0; ih<h.n_elem; ih++) {

    // call stepper
    cout << "  h = " << h(ih) << ":";
    mat Y = ERK2.Evolve(tspan, h(ih), y0);

    // output solution, errors, and overall error
    mat Yerr = abs(Y-Ytrue);
    e(ih) = Yerr.max();
    if (ih > 0) {
      cout << "  Max error = " << e(ih) << ",  conv rate = " 
           << log(e(ih)/e(ih-1))/log(h(ih)/h(ih-1)) << endl;
    } else {
      cout << "  Max error = " << e(ih) << endl;
    }

  }


  //---- ERK3 ----
  cout << "\nERK3:\n";
  ERK3Stepper ERK3(f, y0);
  for (size_t ih=0; ih<h.n_elem; ih++) {

    // call stepper
    cout << "  h = " << h(ih) << ":";
    mat Y = ERK3.Evolve(tspan, h(ih), y0);

    // output solution, errors, and overall error
    mat Yerr = abs(Y-Ytrue);
    e(ih) = Yerr.max();
    if (ih > 0) {
      cout << "  Max error = " << e(ih) << ",  conv rate = " 
           << log(e(ih)/e(ih-1))/log(h(ih)/h(ih-1)) << endl;
    } else {
      cout << "  Max error = " << e(ih) << endl;
    }

  }


  //---- ERK4 ----
  cout << "\nERK4:\n";
  ERK4Stepper ERK4(f, y0);
  for (size_t ih=0; ih<h.n_elem; ih++) {

    // call stepper
    cout << "  h = " << h(ih) << ":";
    mat Y = ERK4.Evolve(tspan, h(ih), y0);

    // output solution, errors, and overall error
    mat Yerr = abs(Y-Ytrue);
    e(ih) = Yerr.max();
    if (ih > 0) {
      cout << "  Max error = " << e(ih) << ",  conv rate = " 
           << log(e(ih)/e(ih-1))/log(h(ih)/h(ih-1)) << endl;
    } else {
      cout << "  Max error = " << e(ih) << endl;
    }

  }


  return 0;
}
