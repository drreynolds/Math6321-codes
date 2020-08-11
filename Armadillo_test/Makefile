CXX=g++
CXXFLAGS=-O --std=c++11

armadillo_test.exe : armadillo_test.cpp GramSchmidt.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ -larmadillo

clean :
	\rm a_data e.txt tmp.txt
