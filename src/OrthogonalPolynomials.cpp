#include "clf/OrthogonalPolynomials.hpp"

#include <assert.h>

using namespace clf;

OrthogonalPolynomials::OrthogonalPolynomials() : 
  BasisFunctions() {}

double OrthogonalPolynomials::Evaluate(int const order, double const x) const {
  assert(order>=0);

  if( order==0 ) { return Phi0(); }
  if( order==1 ) { return Phi1(x); }

  // compute the orthogonal polynomial with the downward Clenshaw algorithm
  double yp2 = 0.0, yp1 = 0.0, yp = 1.0;

  for( int p=order-1; p>=0; --p ) {
    yp2 = yp1; yp1 = yp;

    const double alpha = Ap(p+1)*x + Bp(p+1);
    const double beta = -Cp(p+2);
    yp = alpha*yp1 + beta*yp2;
  }
  
  const double beta = -Cp(2);

  return yp1*Phi1(x) + beta*Phi0()*yp2;
}

Eigen::VectorXd OrthogonalPolynomials::EvaluateAll(int const order, double const x) const {
  Eigen::VectorXd eval(order+1);
  eval(0) = Phi0();
  if( order>0 ) { eval(1) = Phi1(x); }

  for( int p=2; p<=order; ++p ) { eval(p) = (Ap(p)*x+Bp(p))*eval(p-1) - Cp(p)*eval(p-2); }

  return eval;
}

double OrthogonalPolynomials::EvaluateDerivative(int const p, double const x, std::size_t const k) const {
  if( p==0 | k>p ) { return 0.0; }
  if( p==1 ) { 
    assert(k==1);
    return dPhi1dx();
  }

  const double A = Ap(p);

  return k*A*(k==1? Evaluate(p-1, x) : EvaluateDerivative(p-1, x, k-1)) + (A*x+Bp(p))*EvaluateDerivative(p-1, x, k) - Cp(p)*EvaluateDerivative(p-2, x, k);
}

Eigen::MatrixXd OrthogonalPolynomials::EvaluateAllDerivatives(int const p, double const x, std::size_t const k) const {
  if( k==0 ) { return Eigen::MatrixXd(); }
  
  Eigen::MatrixXd eval = Eigen::MatrixXd::Zero(p+1, k);
  if( p>0 ) { eval(1, 0) = 1.0; }

  for( int i=2; i<=p; ++i ) {
    const double A = Ap(i), B = Bp(i), C = Cp(i);
    for( std::size_t d=0; d<std::min((std::size_t)i, k); ++d ) { eval(i, d) = (d+1)*A*(d==0? Evaluate(i-1, x) : eval(i-1, d-1)) + (A*x+B)*eval(i-1, d) - C*eval(i-2, d); }
  }

  return eval;
}
