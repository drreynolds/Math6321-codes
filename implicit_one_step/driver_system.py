#!/usr/bin/env python3
#
# Main routine to test the forward Euler method on a system of ODEs
#    y' = f(t,y), t in [0,1],
#    y(0) = y0.
#
# D.R. Reynolds
# Math 6321 @ SMU
# Fall 2023
import numpy as np
import sys
sys.path.append('..')
from shared.ImplicitSolver import *
from DIRK import *
from IRK import *

# get problem size from command line, otherwise set to 5
N = 5
if (len(sys.argv) > 1):
    N = int(sys.argv[1])
print("\nRunning system ODE problem with N = ", N)

# set up problem data
V = np.eye(N) + np.random.random_sample((N,N))  # fill V,D with random numbers
D = np.diag(-np.random.random_sample(N))
Vinv = np.linalg.inv(V)                         # Vinv = V^{-1}
A = V @ D @ Vinv                                # construct system matrix
if (N < 10):
    print("\nProblem-defining matrices:")
    print("V:\n", V)
    print("Vinv:\n", Vinv)
    print("D:\n", D)
    print("A:\n", A)

# set problem time interval and initial condition
t0 = 0.0
tf = 1.0
y0 = np.random.random_sample(N)

# flag to switch between dense and iterative linear solvers
iterative = False

# problem-defining functions
def f(t,y):
    """ ODE RHS function """
    return A@y
def J(t,y):
    """ Jacobian (in dense matrix format) of the right-hand side function, J(t,y) = df/dy """
    return A
def Jv(t,y,v):
    """ Jacobian-vector-product of the right-hand side function, J(t,y) = (df/dy)@v """
    return A@v
def ytrue(t):
    """ Analytical solution """
    eD = np.zeros((N,N))       # construct the matrix exponential
    for i in range(N):
        eD[i,i] = np.exp(D[i,i]*(t-t0))
    return (V @ (eD @ (Vinv @ y0)))  # ytrue = V exp(D*t) V^{-1} y0

# construct implicit solver
if (iterative):
    solver = ImplicitSolver(Jv, solver_type='gmres', maxiter=20, rtol=1e-12, atol=1e-14)
else:
    solver = ImplicitSolver(J, solver_type='dense', maxiter=20, rtol=1e-12, atol=1e-14, Jfreq=2)

# shared testing data
Nout = 3   # includes initial condition
tspan = np.linspace(t0, tf, Nout)
Ytrue = np.zeros((Nout,N))
for i in range(Nout):
    Ytrue[i,:] = ytrue(tspan[i])

# time steps to try
hvals = 0.5/np.linspace(1,5,5)
errs = np.zeros(hvals.size)

# test runner function
def RunTest(stepper, name):

    print("\n", name, " tests:", sep='')
    for idx, h in enumerate(hvals):
        print("    h = %.3f:" % (h), sep='', end='')
        stepper.reset()
        stepper.sol.reset()
        Y, success = stepper.Evolve(tspan, y0, h)
        Yerr = np.abs(Y-Ytrue)
        errs[idx] = np.linalg.norm(Yerr,np.inf)
        if (success):
            print("  solves = %4i  Niters = %6i  NJevals = %5i  abserr = %8.2e" %
                  (stepper.get_num_solves(), stepper.sol.get_total_iters(),
                   stepper.sol.get_total_setups(), errs[idx]))
    orders = np.log(errs[0:-2]/errs[1:-1])/np.log(hvals[0:-2]/hvals[1:-1])
    print('    estimated order:  max = %.2f,  avg = %.2f' %
          (np.max(orders), np.average(orders)))


# RadauIIA2 tests
A_, b_, c_, p_ = RadauIIA2()
RIIA2 = IRK(f, solver, A_, b_, c_)
RunTest(RIIA2, 'RadauIIA-2')

# Alexander3 tests
A_, b_, c_, p = Alexander3()
Alex3 = DIRK(f, solver, A_, b_, c_)
RunTest(Alex3, 'Alexander-3')

# Crouzeix & Raviart tests
A_, b_, c_, p = CrouzeixRaviart3()
CR3 = DIRK(f, solver, A_, b_, c_)
RunTest(CR3, 'Crouzeix & Raviart-3')

# Gauss-Legendre-2 tests
A_, b_, c_, p = GaussLegendre2()
GL2 = IRK(f, solver, A_, b_, c_)
RunTest(GL2, 'Gauss-Legendre-2')

# RadauIIA3 tests
A_, b_, c_, p = RadauIIA3()
RIIA3 = IRK(f, solver, A_, b_, c_)
RunTest(RIIA3, 'RadauIIA-3')

# Gauss-Legendre-3 tests
A_, b_, c_, p = GaussLegendre3()
GL3 = IRK(f, solver, A_, b_, c_)
RunTest(GL3, 'Gauss-Legendre-3')

# Gauss-Legendre-6 tests
A_, b_, c_, p = GaussLegendre6()
GL6 = IRK(f, solver, A_, b_, c_)
RunTest(GL6, 'Gauss-Legendre-6')
