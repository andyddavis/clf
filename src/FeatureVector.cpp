#include "clf/FeatureVector.hpp"

using namespace clf;

FeatureVector::FeatureVector(std::shared_ptr<const MultiIndexSet> const& set, std::shared_ptr<BasisFunctions> const& basis) :
  set(set), basis(basis)
{}

std::size_t FeatureVector::NumBasisFunctions() const { return set->NumIndices(); }

Eigen::VectorXd FeatureVector::Evaluate(Eigen::VectorXd const& x) const {
  assert(x.size()==set->Dimension());
 
  std::vector<Eigen::VectorXd> basisEval(set->Dimension());
  for( std::size_t d=0; d<set->Dimension(); ++d ) { basisEval[d] = basis->EvaluateAll(set->MaxIndex(d), x(d)); }

  Eigen::VectorXd output = Eigen::VectorXd::Ones(set->NumIndices());
  for( std::size_t i=0; i<set->NumIndices(); ++i ) {
    for( std::size_t d=0; d<set->Dimension(); ++d ) { output(i) *= basisEval[d](set->indices[i]->alpha[d]); }
  }

  return output;
}
