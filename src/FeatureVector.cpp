#include "clf/FeatureVector.hpp"

using namespace clf;

FeatureVector::FeatureVector(std::shared_ptr<const MultiIndexSet> const& set, std::shared_ptr<BasisFunctions> const& basis, std::shared_ptr<Domain> const& domain) :
  set(set), basis(basis), domain(domain)
{}

std::size_t FeatureVector::InputDimension() const { return set->Dimension(); }

std::size_t FeatureVector::NumBasisFunctions() const { return set->NumIndices(); }

Eigen::VectorXd FeatureVector::Evaluate(Eigen::VectorXd const& x) const {
  assert(x.size()==set->Dimension());

  // evaluate the basis functions
  std::vector<Eigen::VectorXd> basisEval(set->Dimension());
  if( domain ) {
    const Eigen::VectorXd y = domain->MapToHypercube(x);
    for( std::size_t d=0; d<set->Dimension(); ++d ) { basisEval[d] = basis->EvaluateAll(set->MaxIndex(d), y(d)); }
  } else {
    for( std::size_t d=0; d<set->Dimension(); ++d ) { basisEval[d] = basis->EvaluateAll(set->MaxIndex(d), x(d)); }
  }

  // evaluate the feature vector
  Eigen::VectorXd output = Eigen::VectorXd::Ones(set->NumIndices());
  for( std::size_t i=0; i<set->NumIndices(); ++i ) {
    for( std::size_t d=0; d<set->Dimension(); ++d ) { output(i) *= basisEval[d](set->indices[i].alpha[d]); }
  }

  return output;
}
