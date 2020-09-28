# Makefile for simple implicit method examples
#
# Daniel R. Reynolds
# SMU Mathematics
# Fall 2020

# compiler & flags
CXX = g++
#CXXFLAGS = -O -std=c++11
CXXFLAGS = -O0 -g -std=c++11
INCS = -I ../shared -I ../fwd_euler
LIBS = -larmadillo

# executable targets
all : driver.exe

driver.exe : driver.cpp bwd_euler.cpp trapezoidal.cpp ../fwd_euler/fwd_euler.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)


# utilities
clean :
	\rm -rf *.txt *.exe *~ *.dSYM
