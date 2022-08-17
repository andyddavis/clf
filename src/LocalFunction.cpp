#include "clf/LocalFunction.hpp"

using namespace clf;

LocalFunction::LocalFunction(std::shared_ptr<const FeatureMatrix> const& featureMatrix) :
  featureMatrix(featureMatrix)
{}

LocalFunction::LocalFunction(std::shared_ptr<MultiIndexSet> const& set, std::shared_ptr<BasisFunctions> const& basis, std::shared_ptr<Domain> const& domain, std::size_t const outdim) :
  featureMatrix(std::make_shared<FeatureMatrix>(std::make_shared<FeatureVector>(set, basis), outdim, domain))
{
  assert(featureMatrix->domain);
}

LocalFunction::LocalFunction(std::shared_ptr<MultiIndexSet> const& set, std::shared_ptr<BasisFunctions> const& basis, std::shared_ptr<Domain> const& domain, std::shared_ptr<Parameters> const& para) :
  featureMatrix(std::make_shared<FeatureMatrix>(std::make_shared<FeatureVector>(set, basis), para->Get<std::size_t>("OutputDimension"), domain))
{}

std::size_t LocalFunction::InputDimension() const { return featureMatrix->InputDimension(); }

std::size_t LocalFunction::OutputDimension() const { return featureMatrix->numFeatureVectors; }

std::size_t LocalFunction::NumCoefficients() const { return featureMatrix->numBasisFunctions; }

Eigen::VectorXd LocalFunction::SampleDomain() const { return featureMatrix->domain->Sample(); }

std::shared_ptr<Domain> LocalFunction::GetDomain() const { return featureMatrix->domain; }

Eigen::VectorXd LocalFunction::Evaluate(std::shared_ptr<Point> const& x, Eigen::VectorXd const& coeff) const { return Evaluate(x->x, coeff); }

Eigen::VectorXd LocalFunction::Evaluate(Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const { 
  assert(x.size()==InputDimension());
  assert(coeff.size()==NumCoefficients());
  return featureMatrix->ApplyTranspose(x, coeff);
}

Eigen::VectorXd LocalFunction::Derivative(Eigen::VectorXd const& x, Eigen::VectorXd const& coeff, std::shared_ptr<LinearDifferentialOperator> const& linOper) const {
  assert(x.size()==InputDimension());
  assert(coeff.size()==NumCoefficients());
  assert(linOper->outdim==OutputDimension());
  return featureMatrix->ApplyTranspose(x, coeff, linOper);
}
