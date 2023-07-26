#!/usr/bin/env python3
#
# Main routine to test various DIRK and IRK methods on the
# scalar-valued ODE problem
#    y' = lambda*y + (1-lambda)*cos(t) - (1+lambda)*sin(t), t in [0,5],
#    y(0) = 1.
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

# problem time interval and parameters
t0 = 0.0
tf = 5.0
lam = 0.0

# flag to switch between dense and iterative linear solvers
iterative = False

# problem-defining functions
def ytrue(t):
    """ Generates a numpy array containing the true solution to the IVP at a given input t. """
    return np.array([np.sin(t) + np.cos(t)])
def f(t,y):
    """ Right-hand side function, f(t,y), for the IVP """
    return np.array([lam*y[0] + (1.0-lam)*np.cos(t) - (1.0+lam)*np.sin(t)])
def J(t,y):
    """ Jacobian (in dense matrix format) of the right-hand side function, J(t,y) = df/dy """
    return np.array( [ [lam] ] )
def Jv(t,y,v):
    """ Jacobian (in dense matrix format) of the right-hand side function, J(t,y) = df/dy """
    return np.array( [lam*v] )

# construct implicit solver
if (iterative):
    solver = ImplicitSolver(Jv, solver_type='gmres', maxiter=20, rtol=1e-9, atol=1e-12)
else:
    solver = ImplicitSolver(J, solver_type='dense', maxiter=20, rtol=1e-9, atol=1e-12, Jfreq=2)

# shared testing data
Nout = 6   # includes initial condition
tspan = np.linspace(t0, tf, Nout)
Ytrue = np.zeros((Nout, 1))
for i in range(Nout):
    Ytrue[i,:] = ytrue(tspan[i])
y0 = ytrue(t0)
lambdas = np.array( (-1.0, -10.0, -50.0) )
hvals = 1.0 / np.linspace(1, 7, 7)
errs = np.zeros(hvals.size)

# test runner function
def RunTest(stepper, name):

    print("\n", name, " tests:", sep='')
    # loop over stiffness values
    for lam in lambdas:

        # update rhs function, Jacobian, integrators, and implicit solver
        def f(t,y):
            """ Right-hand side function, f(t,y), for the IVP """
            return np.array([lam*y[0] + (1.0-lam)*np.cos(t) - (1.0+lam)*np.sin(t)])
        def J(t,y):
            """ Jacobian (dense) of the right-hand side function, J(t,y) = df/dy """
            return np.array( [ [lam] ] )
        def Jv(t,y,v):
            """ Jacobian-vector product, J(t,y)@v = (df/dy)@v """
            return np.array( [lam*v] )
        stepper.f = f
        if (iterative):
            stepper.sol.f_y = Jv
        else:
            stepper.sol.f_y = J

        print("  lambda = " , lam, ":", sep='')
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
A, b, c, p = RadauIIA2()
RIIA2 = IRK(f, solver, A, b, c)
RunTest(RIIA2, 'RadauIIA-2')

# Alexander3 tests
A, b, c, p = Alexander3()
Alex3 = DIRK(f, solver, A, b, c)
RunTest(Alex3, 'Alexander-3')

# Crouzeix & Raviart tests
A, b, c, p = CrouzeixRaviart3()
CR3 = DIRK(f, solver, A, b, c)
RunTest(CR3, 'Crouzeix & Raviart-3')

# Gauss-Legendre-2 tests
A, b, c, p = GaussLegendre2()
GL2 = IRK(f, solver, A, b, c)
RunTest(GL2, 'Gauss-Legendre-2')

# RadauIIA3 tests
A, b, c, p = RadauIIA3()
RIIA3 = IRK(f, solver, A, b, c)
RunTest(RIIA3, 'RadauIIA-3')

# Gauss-Legendre-3 tests
A, b, c, p = GaussLegendre3()
GL3 = IRK(f, solver, A, b, c)
RunTest(GL3, 'Gauss-Legendre-3')

# Gauss-Legendre-6 tests
A, b, c, p = GaussLegendre6()
GL6 = IRK(f, solver, A, b, c)
RunTest(GL6, 'Gauss-Legendre-6')
