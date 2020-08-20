# Makefile for Newton solver
#
# Daniel R. Reynolds
# SMU Mathematics
# Fall 2020

# compiler & flags
CXX = g++
CXXFLAGS = -O -std=c++11
#CXXFLAGS = -O0 -g -std=c++11
LIBS = -larmadillo


# makefile targets
test_newton.exe : test_newton.cpp newton.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)


# utilities
clean :
	\rm -rf *.txt *.exe *~
