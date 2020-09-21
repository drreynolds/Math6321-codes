# Makefile for forward Euler examples
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
all : driver_fwd_euler.exe driver_fwd_euler_system.exe \
      driver_adapt_euler.exe driver_adapt_euler_system.exe \
			stability_experiment.exe

driver_fwd_euler.exe : driver_fwd_euler.cpp fwd_euler.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

driver_fwd_euler_system.exe : driver_fwd_euler_system.cpp fwd_euler.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

driver_adapt_euler.exe : driver_adapt_euler.cpp adapt_euler.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

driver_adapt_euler_system.exe : driver_adapt_euler_system.cpp adapt_euler.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

stability_experiment.exe : stability_experiment.cpp fwd_euler.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

# utilities
clean :
	\rm -rf *.txt *.exe *~ *.dSYM
