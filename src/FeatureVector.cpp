#include "clf/FeatureVector.hpp"

using namespace clf;

FeatureVector::FeatureVector(std::shared_ptr<const MultiIndexSet> const& set, std::shared_ptr<BasisFunctions> const& basis, std::shared_ptr<Domain> const& domain) :
  set(set), basis(basis), domain(domain)
{}

std::size_t FeatureVector::InputDimension() const { return set->Dimension(); }

std::size_t FeatureVector::NumBasisFunctions() const { return set->NumIndices(); }

std::vector<Eigen::VectorXd> FeatureVector::BasisEvaluation(Eigen::VectorXd const& x) const {
  std::vector<Eigen::VectorXd> basisEval(set->Dimension());
  if( domain ) {
    const Eigen::VectorXd y = domain->MapToHypercube(x);
    for( std::size_t d=0; d<set->Dimension(); ++d ) { basisEval[d] = basis->EvaluateAll(set->MaxIndex(d), y(d)); }
  } else {
    for( std::size_t d=0; d<set->Dimension(); ++d ) { basisEval[d] = basis->EvaluateAll(set->MaxIndex(d), x(d)); }
  }

  return basisEval;
}

std::vector<Eigen::MatrixXd> FeatureVector::BasisDerivatives(Eigen::VectorXd const& x, Eigen::VectorXi const& count) const {
  std::vector<Eigen::MatrixXd> basisDeriv(set->Dimension());
  if( domain ) {
    const Eigen::VectorXd y = domain->MapToHypercube(x);
    for( std::size_t d=0; d<set->Dimension(); ++d ) { basisDeriv[d] = basis->EvaluateAllDerivatives(set->MaxIndex(d), y(d), count(d)); }
  } else {
    for( std::size_t d=0; d<set->Dimension(); ++d ) { basisDeriv[d] = basis->EvaluateAllDerivatives(set->MaxIndex(d), x(d), count(d)); }
  }

  return basisDeriv;
}

Eigen::VectorXd FeatureVector::Evaluate(Eigen::VectorXd const& x) const {
  assert(x.size()==set->Dimension());

  // evaluate the basis functions
  const std::vector<Eigen::VectorXd> basisEval = BasisEvaluation(x);

  // evaluate the feature vector
  Eigen::VectorXd output = Eigen::VectorXd::Ones(set->NumIndices());
  for( std::size_t i=0; i<set->NumIndices(); ++i ) {
    for( std::size_t d=0; d<set->Dimension(); ++d ) { output(i) *= basisEval[d](set->indices[i].alpha[d]); }
  }

  return output;
}

Eigen::VectorXd FeatureVector::Derivative(Eigen::VectorXd const& x, Eigen::VectorXi const& counts, std::optional<Eigen::VectorXd> const& jac) const {
  // we are not taking a derivative, use evaluate becuase it has slightly less overhead
  if( counts.isZero() ) { return Evaluate(x); }

  // evaluate the basis functions and their derivatives
  const std::vector<Eigen::VectorXd> basisEval = BasisEvaluation(x);
  const std::vector<Eigen::MatrixXd> basisDeriv = BasisDerivatives(x, counts);
  
  // evaluate the feature vector derivative
  Eigen::VectorXd output = Eigen::VectorXd::Ones(set->NumIndices());
  for( std::size_t i=0; i<set->NumIndices(); ++i ) {
    for( std::size_t d=0; d<set->Dimension(); ++d ) {
      if( counts(d)==0 ) {
	  output(i) *= basisEval[d](set->indices[i].alpha[d]);
      } else if( jac ) { 
	output(i) *= std::pow((*jac)(d), counts(d))*basisDeriv[d](set->indices[i].alpha[d], counts(d)-1);
      } else {
	output(i) *= basisDeriv[d](set->indices[i].alpha[d], counts(d)-1);
      }
    }
  }
  
  return output;
}

Eigen::VectorXd FeatureVector::Derivative(Eigen::VectorXd const& x, Eigen::VectorXi const& counts) const {
  // we are not taking a derivative, use evaluate becuase it has slightly less overhead
  if( counts.isZero() ) { return Evaluate(x); }

  // the jacobian of the coordinate transformation
  std::optional<Eigen::VectorXd> jac;
  if( domain ) { jac = domain->MapToHypercubeJacobian(); }

  return Derivative(x, counts, jac);
}

