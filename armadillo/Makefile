CXX=g++
CXXFLAGS=-O0 -g --std=c++11

armadillo_test.exe : armadillo_test.cpp GramSchmidt.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ -larmadillo

clean :
	\rm -rf a_data e.txt tmp.txt *.dSYM *~
