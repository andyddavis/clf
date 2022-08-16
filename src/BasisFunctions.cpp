#include "clf/BasisFunctions.hpp"

using namespace clf;

BasisFunctions::BasisFunctions() {}

Eigen::VectorXd BasisFunctions::EvaluateAll(int const p, double const x) const {
  Eigen::VectorXd func(p+1);
  for( int i=0; i<=p; ++i ) { func(i) = Evaluate(i, x); }

  return func;
}

Eigen::MatrixXd BasisFunctions::EvaluateAllDerivatives(int const p, double const x, std::size_t const k) const {
  Eigen::MatrixXd eval(p+1, k);
  for( int i=0; i<=p; ++i ) { for( std::size_t d=1; d<=k; ++d ) { eval(i, d-1) = EvaluateDerivative(i, x, d); } }

  return eval;
}
