/* Main routine to test the generic IRK solver method on the
   scalar-valued ODE problem
     y' = lambda*y + (1-lambda)*cos(t) - (1+lambda)*sin(t), t in [0,5],
     y(0) = 1.

   D.R. Reynolds
   Math 6321 @ SMU
   Fall 2020  */

#include <iostream>
#include "irk.hpp"

using namespace std;
using namespace arma;


// Define classes to compute the ODE RHS function and its Jacobian

//    ODE RHS function class -- instantiates a RHSFunction
class MyRHS: public RHSFunction {
public:
  double lambda;                              // stores some local data
  int Evaluate(double t, vec& y, vec& f) {    // evaluates the RHS function, f(t,y)
    f(0) = lambda*y(0) + (1.0-lambda)*cos(t) - (1.0+lambda)*sin(t);
    return 0;
  }
};

//    ODE RHS Jacobian function class -- instantiates a RHSJacobian
class MyJac: public RHSJacobian {
public:
  double lambda;                              // stores some local data
  int Evaluate(double t, vec& y, mat& J) {    // evaluates the RHS Jacobian, J(t,y)
    J(0,0) = lambda;
    return 0;
  }
};


// Convenience function for analytical solution
vec ytrue(const double t) {
  vec yt(1);
  yt(0) = sin(t) + cos(t);
  return yt;
};

// Convenience function for running tests on a method
void RunTestIRK(IRKStepper& IRK, MyRHS& rhs, MyJac& Jac, vec& lambdas,
                vec& h, vec& e, vec& tspan, vec& y0, mat& Ytrue) {

  // update Newton solver parameters
  IRK.newt.tol = 1e-3;
  IRK.newt.maxit = 20;
  IRK.newt.show_iterates = false;

  // loop over lambda values
  for (int il=0; il<lambdas.n_elem; il++) {

    // set current lambda value into rhs and Jac objects
    cout << "  lambda = " << lambdas(il) << ":\n";
    rhs.lambda = lambdas(il);
    Jac.lambda = lambdas(il);

    // loop over time step sizes
    for (int ih=0; ih<h.n_elem; ih++) {

      // call stepper
      mat Y = IRK.Evolve(tspan, h(ih), y0);

      // output solution, errors, and overall error
      mat Yerr = abs(Y-Ytrue);
      e(ih) = Yerr.max();
      cout << "    h = 1/" << ih+1 << "  steps = " << IRK.nsteps
           << "  NIters = " << IRK.nnewt << "  max err = " << e(ih);
      if (ih > 0) {
        cout << "  conv rate = " << log(e(ih)/e(ih-1))/log(h(ih)/h(ih-1)) << endl;
      } else {
        cout << endl;
      }

    }
    cout << endl;
  }

};


// main routine
int main() {

  // time steps to try
  vec h(7);
  for (size_t i=0; i<7; i++)  h(i) = 1.0/(i+1);

  // storage for errors
  vec e(h.n_elem);

  // lambda values to try
  vec lambdas("-1.0, -10.0, -50.0");

  // set problem information
  vec y0("1.0");
  double t0 = 0.0;
  double Tf = 5.0;

  // set desired output times
  int Nout = 6;  // includes initial condition
  vec tspan = linspace(t0, Tf, Nout);

  // create ODE RHS and Jacobian objects
  MyRHS rhs;
  MyJac Jac;

  // create true solution results
  mat Ytrue(1,Nout);
  for (size_t i=0; i<Nout; i++) {
    Ytrue.col(i) = ytrue(tspan(i));
  }

  //////// RadauIIA 2 stage method -- O(h^3) accurate ////////
  cout << "\nRadauIIA 2-stage IRK method -- O(h^3):\n";

  // create IRK stepper object
  mat RIIA2_A(2,2);
  vec RIIA2_b(2), RIIA2_c(2);
  RIIA2_A(0,0) = 5.0/12.0;
  RIIA2_A(0,1) = -1.0/12.0;
  RIIA2_A(1,0) = 9.0/12.0;
  RIIA2_A(1,1) = 3.0/12.0;
  RIIA2_b(0) = 3.0/4.0;
  RIIA2_b(1) = 1.0/4.0;
  RIIA2_c(0) = 1.0/3.0;
  RIIA2_c(1) = 1.0;
  IRKStepper RIIA2(rhs, Jac, y0, RIIA2_A, RIIA2_b, RIIA2_c);

  // run tests
  RunTestIRK(RIIA2, rhs, Jac, lambdas, h, e, tspan, y0, Ytrue);


  //////// Alexander 3 stage DIRK method -- O(h^3) accurate ////////
  cout << "\nAlexander's 3-stage DIRK method -- O(h^3):\n";

  // create IRK stepper object (replace with DIRK)
  double alpha = 0.43586652150845906;
  double tau2 = 0.5*(1.0+alpha);
  mat Alex3_A(3,3);
  vec Alex3_b(3), Alex3_c(3);
  Alex3_A(0,0) = alpha;
  Alex3_A(1,0) = tau2-alpha;
  Alex3_A(1,1) = alpha;
  Alex3_A(2,0) = -0.25*(6.0*alpha*alpha - 16.0*alpha + 1.0);
  Alex3_A(2,1) = 0.25*(6.0*alpha*alpha - 20*alpha + 5.0);
  Alex3_A(2,2) = alpha;
  Alex3_b(0) = -0.25*(6.0*alpha*alpha - 16.0*alpha + 1.0);
  Alex3_b(1) = 0.25*(6.0*alpha*alpha - 20*alpha + 5.0);
  Alex3_b(2) = alpha;
  Alex3_c(0) = alpha;
  Alex3_c(1) = tau2;
  Alex3_c(2) = 1.0;
  IRKStepper Alex3(rhs, Jac, y0, Alex3_A, Alex3_b, Alex3_c);

  // run tests
  RunTestIRK(Alex3, rhs, Jac, lambdas, h, e, tspan, y0, Ytrue);


  //////// Crouzeix & Raviart 3 stage DIRK method -- O(h^4) accurate ////////
  cout << "\nCrouzeix & Raviart 3-stage DIRK method -- O(h^4):\n";

  // create IRK stepper object (replace with DIRK)
  double gamma = 1.0/sqrt(3.0)*cos(M_PI/18.0) + 0.5;
  double delta = 1.0/(6.0*(2.0*gamma-1.0)*(2.0*gamma-1.0));
  mat CR3_A(3,3);
  CR3_A(0,0) = gamma;
  CR3_A(1,0) = 0.5-gamma;
  CR3_A(1,1) = gamma;
  CR3_A(2,0) = 2.0*gamma;
  CR3_A(2,1) = 1.0-4.0*gamma;
  CR3_A(2,2) = gamma;
  vec CR3_b(3), CR3_c(3);
  CR3_b(0) = delta;
  CR3_b(1) = 1.0-2.0*delta;
  CR3_b(2) = delta;
  CR3_c(0) = gamma;
  CR3_c(1) = 0.5;
  CR3_c(2) = 1.0-gamma;
  IRKStepper CR3(rhs, Jac, y0, CR3_A, CR3_b, CR3_c);

  // run tests
  RunTestIRK(CR3, rhs, Jac, lambdas, h, e, tspan, y0, Ytrue);


  //////// Gauss-Legendre 2 stage method -- O(h^4) accurate ////////
  cout << "\nGauss-Legendre 2 stage IRK method -- O(h^4):\n";

  // create IRK stepper object
  mat GL2_A(2,2);
  vec GL2_b(2), GL2_c(2);
  GL2_A(0,0) = 0.25;
  GL2_A(0,1) = (3.0-2.0*sqrt(3.0))/12.0;
  GL2_A(1,0) = (3.0+2.0*sqrt(3.0))/12.0;
  GL2_A(1,1) = 0.25;
  GL2_b.fill(0.5);
  GL2_c(0) = (3.0 - sqrt(3.0))/6.0;
  GL2_c(1) = (3.0 + sqrt(3.0))/6.0;
  IRKStepper GL2(rhs, Jac, y0, GL2_A, GL2_b, GL2_c);

  // run tests
  RunTestIRK(GL2, rhs, Jac, lambdas, h, e, tspan, y0, Ytrue);


  //////// RadauIIA 3 stage method -- O(h^5) accurate ////////
  cout << "\nRadauIIA 3-stage IRK method -- O(h^5):\n";

  // create IRK stepper object
  mat RIIA3_A(3,3);
  vec RIIA3_b(3), RIIA3_c(3);
  RIIA3_A(0,0) = (88.0 - 7.0*sqrt(6.0))/360.0;
  RIIA3_A(0,1) = (296.0 - 169.0*sqrt(6.0))/1800.0;
  RIIA3_A(0,2) = (-2.0 + 3.0*sqrt(6.0))/225.0;
  RIIA3_A(1,0) = (296.0 + 169.0*sqrt(6.0))/1800.0;
  RIIA3_A(1,1) = (88.0 + 7.0*sqrt(6.0))/360.0;
  RIIA3_A(1,2) = (-2.0 - 3.0*sqrt(6.0))/225.0;
  RIIA3_A(2,0) = (16.0 - sqrt(6.0))/36.0;
  RIIA3_A(2,1) = (16.0 + sqrt(6.0))/36.0;
  RIIA3_A(2,2) = 1.0/9.0;
  RIIA3_b(0) = (16.0 - sqrt(6.0))/36.0;
  RIIA3_b(1) = (16.0 + sqrt(6.0))/36.0;
  RIIA3_b(2) = 1.0/9.0;
  RIIA3_c(0) = (4.0 - sqrt(6.0))/10.0;
  RIIA3_c(1) = (4.0 + sqrt(6.0))/10.0;
  RIIA3_c(2) = 1.0;
  IRKStepper RIIA3(rhs, Jac, y0, RIIA3_A, RIIA3_b, RIIA3_c);

  // run tests
  RunTestIRK(RIIA3, rhs, Jac, lambdas, h, e, tspan, y0, Ytrue);


  //////// Gauss-Legendre 3 stage method -- O(h^6) accurate ////////
  cout << "\nGauss-Legendre 3-stage IRK method -- O(h^6):\n";

  // create IRK stepper object
  mat GL3_A(3,3);
  vec GL3_b(3), GL3_c(3);
  GL3_A(0,0) = 5.0/36.0;;
  GL3_A(0,1) = 2.0/9.0 - sqrt(15.0)/15.0;
  GL3_A(0,2) = 5.0/36.0 - sqrt(15.0)/30.0;
  GL3_A(1,0) = 5.0/36.0 + sqrt(15.0)/24.0;
  GL3_A(1,1) = 2.0/9.0;
  GL3_A(1,2) = 5.0/36.0 - sqrt(15.0)/24.0;
  GL3_A(2,0) = 5.0/36.0 + sqrt(15.0)/30.0;
  GL3_A(2,1) = 2.0/9.0 + sqrt(15.0)/15.0;
  GL3_A(2,2) = 5.0/36.0;
  GL3_b(0) = 5.0/18.0;
  GL3_b(1) = 4.0/9.0;
  GL3_b(2) = 5.0/18.0;
  GL3_c(0) = (5.0 - sqrt(15.0))/10.0;
  GL3_c(1) = 0.5;
  GL3_c(2) = (5.0 + sqrt(15.0))/10.0;
  IRKStepper GL3(rhs, Jac, y0, GL3_A, GL3_b, GL3_c);

  // run tests
  RunTestIRK(GL3, rhs, Jac, lambdas, h, e, tspan, y0, Ytrue);


  //////// Gauss-Legendre 6 stage method -- O(h^12) accurate ////////
  cout << "\nGauss-Legendre 6 stage IRK method -- O(h^12):\n";

  // create IRK stepper object
  mat GL6_A(6,6);
  vec GL6_b(6), GL6_c(6);
  GL6_A(0,0) =  0.042831123094792580851996218950605;
  GL6_A(0,1) = -0.014763725997197424643891429014278;
  GL6_A(0,2) =  0.0093250507064777618411400734121424;
  GL6_A(0,3) = -0.0056688580494835162182488917046817;
  GL6_A(0,4) =  0.0028544333150993149102007359161104;
  GL6_A(0,5) = -0.00081278017126476782600392067714199;
  GL6_A(1,0) =  0.092673491430378856970823740288243;
  GL6_A(1,1) =  0.090190393262034655662118827897123;
  GL6_A(1,2) = -0.020300102293239581308124404430781;
  GL6_A(1,3) =  0.010363156240246421640614877198502;
  GL6_A(1,4) = -0.0048871929280376802268550750181669;
  GL6_A(1,5) =  0.001355561055485051944941864725486;
  GL6_A(2,0) =  0.082247922612843859526233540856659;
  GL6_A(2,1) =  0.19603216233324501065540377853111;
  GL6_A(2,2) =  0.11697848364317276194496135254516;
  GL6_A(2,3) = -0.020482527745656096032756375665715;
  GL6_A(2,4) =  0.007989991899662334513029865501749;
  GL6_A(2,5) = -0.0020756257848663355105554732114538;
  GL6_A(3,0) =  0.087737871974451497214547911112663;
  GL6_A(3,1) =  0.1723907946244069768112077902925;
  GL6_A(3,2) =  0.25443949503200161992267908075603;
  GL6_A(3,3) =  0.11697848364317276194496135254516;
  GL6_A(3,4) = -0.015651375809175699331166122736864;
  GL6_A(3,5) =  0.00341432357674130217775889704455;
  GL6_A(4,0) =  0.084306685134100109759050573175723;
  GL6_A(4,1) =  0.18526797945210699155109273081241;
  GL6_A(4,2) =  0.22359381104609910224930782789182;
  GL6_A(4,3) =  0.2542570695795851051980471095211;
  GL6_A(4,4) =  0.090190393262034655662118827897123;
  GL6_A(4,5) = -0.007011245240793695266831302387034;
  GL6_A(5,0) =  0.086475026360849929529996358578351;
  GL6_A(5,1) =  0.17752635320896999641403691987814;
  GL6_A(5,2) =  0.239625825335829040108171596795;
  GL6_A(5,3) =  0.22463191657986776204878263167818;
  GL6_A(5,4) =  0.19514451252126673596812908480852;
  GL6_A(5,5) =  0.042831123094792580851996218950605;
  GL6_b(0) = 0.085662246189585161703992437901209;
  GL6_b(1) = 0.18038078652406931132423765579425;
  GL6_b(2) = 0.23395696728634552388992270509032;
  GL6_b(3) = 0.23395696728634552388992270509032;
  GL6_b(4) = 0.18038078652406931132423765579425;
  GL6_b(5) = 0.085662246189585161703992437901209;
  GL6_c(0) = 0.0337652428984239749709672651079;
  GL6_c(1) = 0.16939530676686775922945571437594;
  GL6_c(2) = 0.38069040695840154764351126459587;
  GL6_c(3) = 0.61930959304159845235648873540413;
  GL6_c(4) = 0.83060469323313224077054428562406;
  GL6_c(5) = 0.9662347571015760250290327348921;
  IRKStepper GL6(rhs, Jac, y0, GL6_A, GL6_b, GL6_c);

  // run tests
  RunTestIRK(GL6, rhs, Jac, lambdas, h, e, tspan, y0, Ytrue);

  return 0;
}
