# Makefile for bvp examples
#
# Daniel R. Reynolds
# Math 6321 @ SMU
# Fall 2020

# compiler & flags
CXX = g++
#CXXFLAGS = -O -std=c++11
CXXFLAGS = -O0 -g -std=c++11
INCS = -I ../shared
LIBS = -larmadillo
SPARSE = -DUSE_SPARSE
#SPARSE =

# executable targets
all : shooting_driver.exe \
      stencil_driver.exe \
      midpoint_driver.exe \
      #lobatto_driver.exe \
      #hermite_driver.exe

shooting_driver.exe : shooting_driver.cpp adapt_rkf.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

stencil_driver.exe : stencil_driver.cpp
	$(CXX) $(CXXFLAGS) -o $@ $(SPARSE) $^ $(LIBS) $(INCS)

midpoint_driver.exe : midpoint_driver.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

lobatto_driver.exe : lobatto_driver.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

hermite_driver.exe : hermite_driver.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)


# utilities
clean :
	\rm -rf *.txt *.exe *~ *.dSYM
