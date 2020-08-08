CXX=g++ --std=c++11
CXXFLAGS=-DARMA_DONT_USE_WRAPPER
BLASLIB=/usr/local/openblas-0.3.7/gcc-8.3/lib/libopenblas.a
LAPACKLIB=-llapack -lgfortran

armadillo_test.exe : armadillo_test.cpp GramSchmidt.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(BLASLIB) $(LAPACKLIB)

clean :
	\rm a_data e.txt tmp.txt
