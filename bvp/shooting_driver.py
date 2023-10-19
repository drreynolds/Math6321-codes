#!/usr/bin/env python3
#
# Main routine to run a shooting method for solution of a
# second-order, scalar-valued BVP:
#
#    u'' = p(t)*u' + q(t)*u + r(t),  a<t<b,
#    u(a) = ua,  u(b) = ub
#
# where the problem has stiffness that may be adjusted using
# the real-valued parameter lambda<0 [read from the command line]
#
# This driver attempts to solve the problem using a single
# shooting method:
#
# (a) convert BVP to first-order IVP system
# (b) use Newton's method to solve for the shooting parameter u'(0)=s
# (c) within Newton's method, each residual/Jacobian evaluation
#     involves solution of an augmented first-order IVP system; for
#     that we use our adaptive RKF solver.
#
# D.R. Reynolds
# Math 6321 @ SMU
# Fall 2023

import sys
sys.path.append('..')
from shared.ImplicitSolver import *
import numpy as np
from BVP import *
from AdaptRKF import *

# get lambda from the command line, otherwise set to -10
lam = -10.0
if (len(sys.argv) > 1):
    lam = float(sys.argv[1])

# create BVP object
bvp = BVP(lam)

# Define a small 'data' object for the shooting method
# Since both the residual and its Jacobian require evolution of an IVP
# (the Jacobian IVP RHS is the linearized version of that used for the
# residual), and since Newton's method always evaluates the residual
# _before_ the Jacobian, within the residual calculation we evolve one
# augmented system for both the shooting method residual and the two
# augmented IVPs for the Jacobian.  The Jacobian-specific data is then
# stored in this class for reuse by the Jacbian evaluation routine.
class ShootingData:
    def __init__(self):
        self.Y0 = np.zeros(2)
        self.Y1 = np.zeros(2)
sdata = ShootingData()

# Define IVP right-hand side function (evaluates the IVPs for
# both the residual and Jacobian)
def f(t,y):
    return np.array([ y[1],
                      bvp.p(t)*y[1] + bvp.q(t)*y[0] + bvp.r(t),
                      y[3],
                      bvp.p(t)*y[3] + bvp.q(t)*y[2],
                      y[5],
                      bvp.p(t)*y[5] + bvp.q(t)*y[4] ])

# create adaptive RKF solver object
y = np.zeros(6)
rkf = AdaptRKF(f,y)

# Define nonlinear residual and Jacobian functions
def F(c):
    # evolve the augmented IVP
    y = np.array([ c[0], c[1], 1, 0, 0, 1 ])
    Y, success = rkf.Evolve([ bvp.a, bvp.b ], y)

    # store Jacobian-related results in sdata object
    sdata.Y0 = Y[1,2:4]
    sdata.Y1 = Y[1,4:6]

    # evaluate the nonlinear residual: [left BC, right BC]
    res = np.array([ c[0] - bvp.ua, Y[1,0] - bvp.ub ])
    return res

def J(c):
    Jac = np.array([[1, 0], [ sdata.Y0[0], sdata.Y1[0] ]])
    return Jac


# set final solution resolution
N = 1001

# compute/store analytical solution
tspan = np.linspace(bvp.a, bvp.b, N)
utrue = np.zeros(N)
for i in range(N):
    utrue[i] = bvp.utrue(tspan[i])

# since h(c) is linear, then there's no point in running the
# shooting method for various Newton tolerances, so just use one
newt_tol = 1e-3

# loop over various inner IVP tolerances
rkf_rtol = [1.e-5, 1.e-8, 1.e-11]
for rtol in rkf_rtol:

    # set tight IVP tolerances
    atol = rtol/1000
    rkf.set_rtol(rtol)
    rkf.set_atol(atol)

    # output problem information
    print("\nShooting method for BVP with lambda = ", lam, ":")
    print("  newt_tol = ", newt_tol, ",  rtol = ", rtol, ", atol = ", atol)

    # create Newton solver (manually set up)
    newt = ImplicitSolver(J, solver_type='dense', maxiter=20, rtol=newt_tol, Jfreq=1, steady=True)

    # set initial guess
    c0 = np.array([bvp.ua, 0])

    # call Newton solver
    print("  Calling Newton solver:")
    c, iters, success = newt.solve(F, c0)
    if (not success):
        print("    Warning: Newton convergence failure")
    else:
        print("    converged in ", iters, " iterations")

    # output final c value, re-run IVP to generate BVP solution
    print("  Newton solution: ", c)
    y = np.array([c[0], c[1], 0, 0, 0, 0])
    Y, success = rkf.Evolve(tspan, y)

    # output maximum error
    uerr = np.abs(Y[:,0]-utrue)
    print("  Maximum BVP solution error = %.4e" % (np.max(uerr)))
