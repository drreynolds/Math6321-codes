# Makefile for explicit one-step method examples
#
# Daniel R. Reynolds
# SMU Mathematics
# Fall 2020

# compiler & flags
CXX = g++
#CXXFLAGS = -O -std=c++11
CXXFLAGS = -O0 -g -std=c++11
INCS = -I ../shared
LIBS = -larmadillo

# makefile targets
all : driver.exe

driver.exe : driver.o taylor2.o erk2.o heun.o erk3.o erk4.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $^ $(INCS)

# utilities
clean :
	\rm -rf *.o *.txt *.exe *~ *.dSYM
