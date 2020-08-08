/* Daniel R. Reynolds
   SMU Mathematics
   7 August 2020 */

// Inclusions
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <armadillo>
using namespace std;


// Gram-Schmidt process for orthonormalizing a set of vectors
int GramSchmidt(arma::mat& X) {

  // check that there is work to do
  if (X.n_cols < 1)  return 0;

  // get entry magnitude (for linear dependence check)
  double Xmax = arma::norm(X,"inf");

  // normalize first column
  double colnorm = arma::norm(X.col(0));
  if (colnorm < 1.e-13*Xmax) {
    cerr << "GramSchmidt error: vectors are linearly-dependent!\n";
    return 1;
  }
  X.col(0) *= (1.0/colnorm);

  // iterate over remaining vectors, performing Gram-Schmidt process
  for (int i=1; i<X.n_cols; i++) {

    // subtract off portions in directions of existing basis vectors
    for (int j=0; j<i; j++)
      X.col(i) -= (arma::dot(X.col(i), X.col(j)) * X.col(j));

    // normalize vector, checking for linear dependence
    colnorm = arma::norm(X.col(i));
    if (colnorm < 1.e-13*Xmax) {
      cerr << "GramSchmidt error: vectors are linearly-dependent!\n";
      return 1;
    }
    X.col(i) *= (1.0/colnorm);
  }

  // return success
  return 0;
}
