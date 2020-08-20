# Makefile for I/O and plotting example
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
all : io_intro.exe

io_intro.exe : io_intro.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

clean :
	\rm -f *.o x.txt T*.txt *.png *.pdf

realclean : clean
	\rm -f *.exe *~


####### End of Makefile #######
