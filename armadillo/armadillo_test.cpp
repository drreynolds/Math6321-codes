/* Daniel R. Reynolds
   SMU Mathematics
   6 August 2020 */

// Inclusions
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <armadillo>
using namespace std;

// prototypes of other functions
int GramSchmidt(arma::mat& X);


// Example routine to test the Mat class
int main(int argc, char* argv[]) {

  // create a row vector of length 5
  arma::rowvec a(5);
  a.fill(0.0);

  // create a row vec with an existing data array
  double dat1[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
  arma::rowvec b(dat1, 5);

  // create a column vec with an existing vector
  double dat2[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
  arma::vec b2(dat2, 5);

  // create a row vector using linspace
  arma::rowvec c = arma::linspace<arma::rowvec>(1.0, 5.0, 5);

  // create a column vector using the single integer constructor
  arma::vec h(7);

  // output vectors above to screen
  cout << "writing array of zeros:\n";
  cout << a << endl;
  cout << "writing array of 0.1,0.2,0.3,0.4,0.5:\n";
  cout << b << endl;
  cout << "writing (column) array of 0.1,0.2,0.3,0.4,0.5:\n";
  cout << b2 << endl;
  cout << "writing array of 1,2,3,4,5:\n";
  cout << c << endl;
  cout << "writing a column vector of 7 zeros:\n";
  cout << h << endl;

  // verify that b has size 5
  if (b.n_elem != 5)
    cerr << "error: incorrect matrix size\n";
  if (b.n_cols != 5)
    cerr << "error: incorrect matrix columns\n";
  if (b.n_rows != 1)
    cerr << "error: incorrect matrix rows\n";

  // edit entries of a in both matrix forms, and write each entry of a to screen
  a(0)  = 10.0;
  a(1)  = 15.0;
  a[2] = 20.0;
  a(0,3) = 25.0;
  a.at(4) = 30.0;
  cout << "entries of a, one at a time: should give 10, 15, 20, 25, 30\n";
  for (size_t i=0; i<a.n_elem; i++)
    cout << "  " << a[i] << endl;

  // write the values to file
  cout << "writing this same vector to the file 'a_data':\n";
  a.save("a_data", arma::raw_ascii);

  // Testing MatrixRead() constructor
  double tol = 2.0e-15;
  arma::mat read_test1a = arma::randu(3,4);
  read_test1a.save("tmp.txt", arma::raw_ascii);
  arma::mat read_test1b;
  read_test1b.load("tmp.txt");
  arma::mat read_test1_error = read_test1a - read_test1b;
  if (norm(read_test1_error,"inf") < tol)
    cout << "save/load test 1 passed\n";
  else {
    cout << "save/load test 1 failed, ||error|| = " << norm(read_test1_error,"inf") << endl;
    cout << "  read_test1a = \n" << read_test1a << endl;
    cout << "  read_test1b = \n" << read_test1b << endl;
  }

  arma::mat read_test2a = arma::randu(12,1);
  read_test2a.save("tmp.txt", arma::raw_ascii);
  arma::mat read_test2b;
  read_test2b.load("tmp.txt");
  arma::mat read_test2_error = read_test2a - read_test2b;
  if (norm(read_test2_error,"inf") < tol)
    cout << "save/load test 2 passed\n";
  else {
    cout << "save/load test 2 failed, ||error|| = " << norm(read_test2_error,"inf") << endl;
    cout << "  read_test2a = \n" << read_test2a << endl;
    cout << "  read_test2b = \n" << read_test2b << endl;
  }

  arma::mat read_test3a = arma::randu(1,7);
  read_test3a.save("tmp.txt", arma::raw_ascii);
  arma::mat read_test3b;
  read_test3b.load("tmp.txt");
  arma::mat read_test3_error = read_test3a - read_test3b;
  if (norm(read_test3_error,"inf") < tol)
    cout << "save/load test 3 passed\n";
  else {
    cout << "save/load test 3 failed, ||error|| = " << norm(read_test3_error,"inf") << endl;
    cout << "  read_test3a = \n" << read_test3a << endl;
    cout << "  read_test3b = \n" << read_test3b << endl;
  }

  // Testing copy constructor
  arma::mat B = a;
  cout << "arma::mat B = a uses copy constructor, should give 10, 15, 20, 25, 30\n";
  cout << B << endl;
  // update one entry of a
  cout << "updating the 5th entry of a to be 31:\n";
  a(4) = 31.0;
  cout << "   a = " << a << endl;

  cout << "B should not have changed" << endl;
  cout << "   B = " << B << endl;
  a(4) = 30.0;  // reset to original

  // Testing submatrix copy constructor
  arma::mat B2 = a.submat(0,1,0,3);  // B2 = a(0:0,1:3)
  cout << "arma::mat B2 = a.submat(0,1,0,3) uses submatrix copy constructor" << endl;
  cout << B2 << endl;
  // update entries of B2
  B2(0) = 4.0;
  B2(1) = 3.0;
  B2(2) = 2.0;
  // copy B2 back into a using submatrix copy
  a(0,arma::span(1,3)) = B2;  // a(0,1:3) = B2
  cout << "span copy back into a, should have entries 10 4 3 2 30" << endl;
  cout << a << endl;
  a(1) = 15.0;  // reset to original
  a(2) = 20.0;
  a(3) = 25.0;

  // Test arithmetic operators
  cout << "Testing vector add, should give 1.1, 2.2, 3.3, 4.4, 5.5\n";
  b += c;  // b = b + c
  cout << b << endl;

  cout << "Testing scalar add, should give 2, 3, 4, 5, 6\n";
  c += 1.0;  // c = c + 1
  cout << c << endl;

  cout << "Testing vector subtract, should be 8, 12, 16, 20, 24\n";
  a -= c;  // a = a - c
  cout << a << endl;

  cout << "Testing scalar subtract, should be 0, 1, 2, 3, 4\n";
  c -= 2.0;  // c = c - 2
  cout << c << endl;

  cout << "Testing vector fill, should all be -1\n";
  b.fill(-1.0);  // b = -1*ones(size(b))
  cout << b << endl;

  cout << "Testing vector copy, should be 0, 1, 2, 3, 4\n";
  a = c;
  cout << a << endl;

  cout << "Testing scalar multiply, should be 0, 5, 10, 15, 20\n";
  c *= 5.0;  // c = c * 5
  cout << c << endl;

  cout << "Testing deep copy, should be 0, 1, 2, 3, 4\n";
  cout << a << endl;

  cout << "Testing vector multiply, should be 0, -1, -2, -3, -4\n";
  b %= a;   // b = b.*a
  cout << b << endl;

  cout << "Testing vector divide, should be 0, -2.5, -3.3333, -3.75, -4\n";
  arma::mat j(c);  // j = c
  b += -1.0;
  j /= b;   // j = j ./ b
  b += 1.0;
  cout << j << endl;

  cout << "Testing vector +=, should be 0, 4, 8, 12, 16\n";
  b += c;
  cout << b << endl;

  cout << "Testing scalar +=, should be 1, 6, 11, 16, 21\n";
  c += 1.0;
  cout << c << endl;

  cout << "Testing vector -=, should be 1, 2, 3, 4, 5\n";
  c -= b;
  cout << c << endl;

  cout << "Testing scalar -=, should be -2, -1, 0, 1, 2\n";
  a -= 2.0;
  cout << a << endl;

  cout << "Testing vector %=, should be 0, -4, 0, 12, 32\n";
  a %= b;
  cout << a << endl;

  cout << "Testing scalar *=, should be 2, 4, 6, 8, 10\n";
  c *= 2.0;
  cout << c << endl;

  cout << "Testing vector /=, should be 0, -1, 0, 1.5, 3.2\n";
  j = a;
  j /= c;
  cout << j << endl;

  cout << "Testing scalar /=, should be 1, 2, 3, 4, 5\n";
  j = c;
  j /= 2.0;
  cout << j << endl;

  cout << "Testing vector =, should be 2, 4, 6, 8, 10\n";
  b = c;
  cout << b << endl;

  cout << "Testing scalar fill, should be 3, 3, 3, 3, 3\n";
  a.fill(3.0);
  cout << a << endl;

  cout << "Testing vector norm, should be 14.8324\n";
  cout << "  " << arma::norm(b) << endl;

  cout << "Testing vector infinity norm, should be 10\n";
  cout << "  " << arma::norm(b,"inf") << endl;

  cout << "Testing vector one norm, should be 30\n";
  cout << "  " << arma::norm(b,1) << endl;

  cout << "Testing vector min, should be 2\n";
  cout << "  " << b.min() << endl;

  cout << "Testing vector max, should be 10\n";
  cout << "  " << b.max() << endl;

  B = arma::mat(2,5);
  B(0,arma::span(0,4)) = c;   // B(0,:) = c
  B(1,arma::span(0,4)) = a;   // B(1,:) = a
  B += 2.0;

  cout << "Testing matrix infinity norm, should be 40\n";
  cout << "  " << arma::norm(B,"inf") << endl;

  cout << "Testing matrix one norm, should be 17\n";
  cout << "  " << arma::norm(B,1) << endl;

  cout << "Testing matrix two norm, should be 21.7821\n";
  cout << "  " << arma::norm(B,2) << endl;

  cout << "Testing matrix min, should be 4\n";
  cout << "  " << B.min() << endl;

  cout << "Testing matrix max, should be 12\n";
  cout << "  " << B.max() << endl;

  cout << "Testing dot, should be 90\n";
  cout << "  " << dot(a, c) << endl;

  cout << "Testing logspace, should be 0.01 0.1 1 10 100\n";
  arma::mat e = arma::logspace<arma::rowvec>(-2.0, 2.0, 5);
  cout << e << endl;

  ofstream out;
  out.open("e.txt");
  if(out.is_open())
  {
    out << e;
    cout << "Wrote to file e.txt:\n" << e;
  }
  out.close();

  cout << "Testing randu\n";
  arma::mat f = arma::randu(3,3);
  cout << "f = " << f << endl;
  cout << "Testing write with a temporary result" << endl;
  cout << (f*f+f) << endl;
  cout << "f should be unchanged from above" << endl;
  cout << "f = " << f << endl;
  cout << "Testing f==f, should be 3x3 matrix of ones\n" << (f==f) << endl;
  b.fill(1.0);
  cout << "Testing e==b, should be 0 0 1 0 0\n  " << (e==b) << endl;

  // create and fill in a 10x5 matrix
  arma::mat Y(10,5);
  for (size_t i=0; i<10; i++) {
    Y(i,0) = 1.0*i;
    Y(i,1) = -5.0 + 1.0*i;
    Y(i,2) = 2.0 + 2.0*i;
    Y(i,3) = 20.0 - 1.0*i;
    Y(i,4) = -20.0 + 1.0*i;
  }

  // extract columns from matrix (both ways)
  arma::vec Y0 = Y.col(0);
  arma::mat Y1(Y.col(1));
  arma::vec Y2(Y(arma::span(0,9),2));
  arma::vec Y3 = Y.col(3);
  arma::vec Y4 = Y(arma::span(0,9),4);

  // check the LinearSum routine
  Y4 += Y3;
  cout << "Testing column extraction, should be all zeros:\n";
  cout << Y4 << endl;

  // check linear sum 
  arma::mat d = arma::linspace<arma::rowvec>(0.0, 4.0, 5);
  cout << "Testing LinearSum, should be 0.02 1.2 4 23 204:\n";
  arma::mat g = 1.0*d + 2.0*e;
  cout << g << endl;

  // check the pow routine
  d = arma::pow(d,2.0);   // d = d.^2
  cout << "Testing pow, should be 0 1 4 9 16:\n";
  cout << d << endl;
  d = arma::pow(d,0.5);   // d = sqrt(d)
  cout << "Testing pow, should be 0 1 2 3 4:\n";
  cout << d << endl;

  // check the abs routine
  Y1 = arma::abs(Y1);
  cout << "Testing abs, should be the column 5 4 3 2 1 0 1 2 3 4:\n";
  cout << Y1 << endl;

  // check the inplace_trans routine
  cout << "Testing inplace_trans, should be the row 5 4 3 2 1 0 1 2 3 4:\n";
  inplace_trans(Y1);
  cout << Y1 << endl;

  // check the copy-based transpose routine
  cout << "Testing copy-based transpose, should be the column 5 4 3 2 1 0 1 2 3 4:\n";
  Y2 = Y1.t();
  cout << Y2 << endl;

  cout << "Testing GramSchmidt, should work\n";
  arma::mat X = arma::randu(20,3);
  int iret = GramSchmidt(X);
  cout << "  GramSchmidt returned " << iret << ", dot-products are:\n";
  cout << "     <X0,X0> = " << arma::dot(X.col(0),X.col(0)) << endl;
  cout << "     <X0,X1> = " << arma::dot(X.col(0),X.col(1)) << endl;
  cout << "     <X0,X2> = " << arma::dot(X.col(0),X.col(2)) << endl;
  cout << "     <X1,X1> = " << arma::dot(X.col(1),X.col(1)) << endl;
  cout << "     <X1,X2> = " << arma::dot(X.col(1),X.col(2)) << endl;
  cout << "     <X2,X2> = " << arma::dot(X.col(2),X.col(2)) << endl << endl;

  cout << "Testing GramSchmidt, should fail\n";
  arma::mat V = arma::randu(20,3);
  V.col(2) = 2.0*V.col(1);
  iret = GramSchmidt(V);
  cout << "  GramSchmidt returned " << iret << ", dot-products are:\n";
  cout << "     <V0,V0> = " << arma::dot(V.col(0),V.col(0)) << endl;
  cout << "     <V0,V1> = " << arma::dot(V.col(0),V.col(1)) << endl;
  cout << "     <V0,V2> = " << arma::dot(V.col(0),V.col(2)) << endl;
  cout << "     <V1,V1> = " << arma::dot(V.col(1),V.col(1)) << endl;
  cout << "     <V1,V2> = " << arma::dot(V.col(1),V.col(2)) << endl;
  cout << "     <V2,V2> = " << arma::dot(V.col(2),V.col(2)) << endl << endl;

  cout << "Testing matrix product, should be: 9 -1 9 -8 11 6\n";
  arma::mat A_ = arma::eye(6,6);
  A_(0,3) = 2.0;
  A_(1,2) = -1.0;
  A_(2,5) = 1.0;
  A_(3,5) = -2.0;
  A_(4,5) = 1.0;
  arma::mat xtrue_ = arma::linspace(1.0, 6.0, 6);
  arma::mat b_ = A_*xtrue_;
  cout << b_ << endl;

  cout << "Testing backwards substitution solve with provided solution array:\n";
  arma::vec x_(6);
  if (arma::solve(x_, A_, b_)) {
    cout << "  solve succeeded\n";
  } else {
    cout << "  solve failed\n";
  }
  cout << "  ||x - xtrue|| = " << arma::norm(x_ - xtrue_, "inf") << "\n\n";

  cout << "Testing forwards substitution with provided solution array:\n";
  A_.eye();
  A_(3,0) = 2.0;
  A_(2,1) = -1.0;
  A_(5,2) = 1.0;
  A_(5,3) = -2.0;
  A_(5,4) = 1.0;
  b_ = A_*xtrue_;
  x_ = 0.0;
  if (arma::solve(x_, A_, b_)) {
    cout << "  solve succeeded\n";
  } else {
    cout << "  solve failed\n";
  }
  cout << "  ||x - xtrue|| = " << arma::norm(x_ - xtrue_, "inf") << "\n\n";

  cout << "Testing general solver:\n";
  arma::mat C_ = 100.0*arma::eye(9,9) + arma::randu(9,9);
  arma::vec z_ = arma::logspace(-4.0, 4.0, 9);
  arma::mat f_ = C_*z_;
  arma::mat g_ = arma::solve(C_, f_);
  cout << "  ||x - xtrue|| = " << arma::norm(g_ - z_, "inf") << "\n\n";

  cout << "Testing copy-into-col, should be: \n";
  cout << "    0.01   0.02   0.04   0.08\n";
  cout << "    0.1    0.2    0.4    0.8\n";
  cout << "    1      2      4      8\n";
  cout << "   10     20     40     80\n";
  cout << " Actually is:\n";
  arma::mat z2_ = arma::logspace(-2.0, 1.0, 4);
  arma::mat B_(4,4);
  B_.col(0) = z2_;
  z2_ *= 2.0;
  B_.col(1) = z2_;
  z2_ *= 2.0;
  B_.col(2) = z2_;
  z2_ *= 2.0;
  B_.col(3) = z2_;
  cout << B_ << endl;

  cout << "Testing general solver with matrix-valued rhs:\n";
  arma::mat E_ = 100.0*arma::eye(4,4) + arma::randu(4,4);
  arma::mat F_ = E_*B_;
  arma::mat X_ = arma::solve(E_, F_);
  cout << "  ||X - Xtrue|| = " << arma::norm(X_ - B_,"inf") << "\n\n";

  cout << "Testing matrix inverse:\n";
  arma::mat D_ = 10.0*arma::eye(8,8) + arma::randu(8,8);
  arma::mat DDinv_(D_);
  arma::mat Dinv_ = D_.i();
  DDinv_ = D_*Dinv_;
  cout << "  ||I - D*Dinv|| = " << arma::norm(arma::eye(8,8) - DDinv_,"inf") << endl;
  DDinv_ = Dinv_*D_;
  cout << "  ||I - Dinv*D|| = " << arma::norm(arma::eye(8,8) - DDinv_,"inf") << endl;

  return 0;
} // end main
