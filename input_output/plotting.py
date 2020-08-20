#!/usr/bin/env python
#
# I/O and Plotting Introduction Python Script
#
# D.R. Reynolds
# Math 6321 @ SMU
# Fall 2020

# import student code
from numpy import *
from matplotlib.pyplot import *

# First, load our saved data files
x = loadtxt('x.txt')
T = loadtxt('T.txt')

# plot similarly to Matlab; only "legend" command differs
# note that all text fields may include $$ for LaTeX rendering
figure(1)
plot(x,T)
xlabel('$x$')
ylabel('$y$')
title('Chebyshev polynomials')
legend(('$T_1(x)$', '$T_3(x)$', '$T_5(x)$', '$T_7(x)$', '$T_9(x)$'))
savefig('figure1.png')

# we can also plot with manual colors and line styles, using 
# identical formatting specifiers as in Matlab
figure(2)
plot(x, T[:,0], 'b-',  label='$T_1(x)$')
plot(x, T[:,1], 'r--', label='$T_3(x)$')
plot(x, T[:,2], 'm:',  label='$T_5(x)$')
plot(x, T[:,3], 'g-.', label='$T_7(x)$')
plot(x, T[:,4], 'c-', label='$T_9(x)$')
xlabel('$x$')
ylabel('$y$')
title('Chebyshev polynomials')
legend()
savefig('figure2.pdf')

# display all plots; these can be interacted with using the mouse
show()
