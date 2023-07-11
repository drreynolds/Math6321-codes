# Math6321-codes

Codes for in-class collaboration for the course: Numerical Solution of ODEs (MATH 6321) at Southern Methodist University, for the Fall 2023 semester.

These codes require a modern Python installation.

   *Note: the `c++` branch includes implementations of the same solvers, but in C++, and that use the "Armadillo" C++ library (http://arma.sourceforge.net) for vectors, matrices, and linear solvers.*

Codes are grouped according to type:

* `initial_demo` -- simple demonstration scripts showing the use of Python for mathematical calculations and plotting.
* `forward_euler` -- simple IVP "evolution" routine, based on the simplest IVP solver.  Basic approach for timestep adaptivity.  Contains two classes, `ForwardEuler` (fixed-step evolution) and `AdaptEuler` (adaptive-step evolution).
* `shared` -- reusable `ImplicitSolver` class, to be used by implicit ODE methods.
* `newton` -- test driver to show use of `ImplicitSolver`.
* `simple_implicit` -- simple implicit ODE solver classes, `BackwardEuler` and `Trapezoidal`, showing use of the `ImplicitSolver` class for implicit ODE methods.
* `explicit_one_step` -- higher-order explicit, one-step, ODE integration methods, containing the `Taylor2` and `ERK` classes.
* `implicit_one_step` -- higher-order implicit, one-step, ODE integration methods, containing the `DIRK` and `IRK` classes.
* `linear_multistep` -- higher-order explicit and implicit multi-step ODE integration methods, containing the classes `ExplicitLMM` and `ImplicitLMM`.
* `bvp` -- two-point boundary-value problem solvers.

Daniel R. Reynolds  
Mathematics @ SMU  
