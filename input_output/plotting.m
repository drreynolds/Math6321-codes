% I/O and Plotting Introduction Matlab Script
%
% D.R. Reynolds
% Math 6321 @ SMU
% Fall 2020

% First, load our saved data files
x = load('x.txt');
T = load('T.txt');

% check that these were properly loaded
whos

% plot as usual
plot(x,T)
xlabel('x')
ylabel('y')
title('Chebyshev polynomials')
legend('T_1(x)','T_3(x)','T_5(x)','T_7(x)','T_9(x)')

% end of script
