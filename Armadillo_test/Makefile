CXX=g++
CXXFLAGS=-O --std=c++11
LIBS=-larmadillo

armadillo_test.exe : armadillo_test.cpp GramSchmidt.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

clean :
	\rm a_data e.txt tmp.txt
