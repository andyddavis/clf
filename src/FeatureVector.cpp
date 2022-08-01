#include "clf/FeatureVector.hpp"

using namespace clf;

FeatureVector::FeatureVector(std::shared_ptr<const MultiIndexSet> const& set, std::shared_ptr<BasisFunctions> const& basis, Eigen::VectorXd const& xbar, double const delta) :
  set(set), basis(basis), xbar(xbar), delta(delta)
{
  assert(xbar.size()==set->Dimension());
}

FeatureVector::FeatureVector(std::shared_ptr<const MultiIndexSet> const& set, std::shared_ptr<BasisFunctions> const& basis, Eigen::VectorXd const& xbar, std::shared_ptr<Parameters> const& para) :
  set(set), basis(basis), xbar(xbar), delta(para->Get<double>("LocalRadius"))
{
  assert(xbar.size()==set->Dimension());
}

std::size_t FeatureVector::InputDimension() const { return set->Dimension(); }

std::size_t FeatureVector::NumBasisFunctions() const { return set->NumIndices(); }

Eigen::VectorXd FeatureVector::Transformation(Eigen::VectorXd const& x) const { return (x-xbar)/delta; }

Eigen::VectorXd FeatureVector::Evaluate(Eigen::VectorXd const& x) const {
  assert(x.size()==set->Dimension());

  const Eigen::VectorXd y = Transformation(x);
 
  std::vector<Eigen::VectorXd> basisEval(set->Dimension());
  for( std::size_t d=0; d<set->Dimension(); ++d ) { basisEval[d] = basis->EvaluateAll(set->MaxIndex(d), y(d)); }

  Eigen::VectorXd output = Eigen::VectorXd::Ones(set->NumIndices());
  for( std::size_t i=0; i<set->NumIndices(); ++i ) {
    for( std::size_t d=0; d<set->Dimension(); ++d ) { output(i) *= basisEval[d](set->indices[i].alpha[d]); }
  }

  return output;
}
