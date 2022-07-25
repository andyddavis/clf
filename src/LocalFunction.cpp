#include "clf/LocalFunction.hpp"

using namespace clf;

LocalFunction::LocalFunction(std::shared_ptr<const FeatureMatrix> const& featureMatrix) :
  featureMatrix(featureMatrix)
{}

std::size_t LocalFunction::InputDimension() const { return featureMatrix->InputDimension(); }

std::size_t LocalFunction::OutputDimension() const { return featureMatrix->numFeatureVectors; }

std::size_t LocalFunction::NumCoefficients() const { return featureMatrix->numBasisFunctions; }

Eigen::VectorXd LocalFunction::Evaluate(Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const { 
  assert(x.size()==InputDimension());
  assert(coeff.size()==NumCoefficients());
  return featureMatrix->ApplyTranspose(x, coeff);
}
