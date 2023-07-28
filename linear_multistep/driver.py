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
from Explicit_LMM import *
#from Implicit_LMM import *

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
    return np.array( [lam*v[0]] )

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
lambdas = np.array( (-1.0, -10.0, -50.0, -1000.0) )
hvals = np.array( (0.1, 0.05, 0.01, 0.005, 0.001) )
errs = np.zeros(hvals.size)

# test runner function
def RunTest(stepper, name, implicit):

    print("\n", name, " tests:", sep='')
    # loop over stiffness values
    for lam in lambdas:

        # update rhs function, Jacobian, integrators, and implicit solver
        def f(t,y):
            """ Right-hand side function, f(t,y), for the IVP """
            return np.array([lam*y[0] + (1.0-lam)*np.cos(t) - (1.0+lam)*np.sin(t)])
        if (implicit):
            def J(t,y):
                """ Jacobian (dense) of the right-hand side function, J(t,y) = df/dy """
                return np.array( [ [lam] ] )
            def Jv(t,y,v):
                """ Jacobian-vector product, J(t,y)@v = (df/dy)@v """
                return np.array( [lam*v[0]] )
            stepper.f = f
            if (iterative):
                stepper.sol.f_y = Jv
            else:
                stepper.sol.f_y = J

        print("  lambda = " , lam, ":", sep='')
        for idx, h in enumerate(hvals):
            print("    h = %.3f:" % (h), sep='', end='')
            stepper.reset()
            if (implicit):
                stepper.sol.reset()
            # create overly-long initial condition vector (sufficient for all methods)
            y0 = np.array([ytrue(t0-5*h), ytrue(t0-4*h), ytrue(t0-3*h), ytrue(t0-2*h),
                           ytrue(t0-h), ytrue(t0)])
            Y, success = stepper.Evolve(tspan, y0, h)
            Yerr = np.abs(Y-Ytrue)
            errs[idx] = np.linalg.norm(Yerr,np.inf)
            if (success):
                if (implicit):
                    print("  solves = %4i  Niters = %6i  NJevals = %5i  abserr = %8.2e" %
                          (stepper.get_num_solves(), stepper.sol.get_total_iters(),
                           stepper.sol.get_total_setups(), errs[idx]))
                else:
                    print("  steps = %4i  Nrhs = %6i  abserr = %8.2e" %
                          (stepper.get_num_steps(), stepper.get_num_rhs(), errs[idx]))
        orders = np.log(errs[0:-2]/errs[1:-1])/np.log(hvals[0:-2]/hvals[1:-1])
        print('    estimated order:  max = %.2f,  avg = %.2f' %
              (np.max(orders), np.average(orders)))


# Adams-Bashforth-1
alphas, betas, p = AdamsBashforth1()
AB1 = Explicit_LMM(f, alphas, betas)
RunTest(AB1, 'Adams-Bashforth-1', False)

# Adams-Bashforth-2
alphas, betas, p = AdamsBashforth2()
AB2 = Explicit_LMM(f, alphas, betas)
RunTest(AB2, 'Adams-Bashforth-2', False)

# Adams-Bashforth-3
alphas, betas, p = AdamsBashforth3()
AB3 = Explicit_LMM(f, alphas, betas)
RunTest(AB3, 'Adams-Bashforth-3', False)

# Adams-Bashforth-4
alphas, betas, p = AdamsBashforth4()
AB4 = Explicit_LMM(f, alphas, betas)
RunTest(AB4, 'Adams-Bashforth-4', False)
