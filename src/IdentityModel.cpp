#include "clf/IdentityModel.hpp"

using namespace clf;

IdentityModel::IdentityModel(std::size_t const indim, std::size_t const outdim, std::shared_ptr<const Parameters> const& para) :
  SystemOfEquations(indim, outdim, para)
{}

Eigen::VectorXd IdentityModel::Operator(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const { 
  assert(u->InputDimension()==indim);
  assert(u->OutputDimension()==outdim);
  assert(u->NumCoefficients()==coeff.size());
  assert(x.size()==indim);
  return u->Evaluate(x, coeff);
}

Eigen::MatrixXd IdentityModel::JacobianWRTCoefficients(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const {
  std::size_t start = 0;
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(outdim, coeff.size());
  for( auto it=u->featureMatrix->Begin(); it!=u->featureMatrix->End(); ++it ) {
  }
  /*for( std::size_t i=0; i<outdim; ++i ) {
    auto vec = u->featureMatrix->GetFeatureVector(i);
    const std::size_t num = vec->NumBasisFunctions();
    jac.row(i).segment(start, num) = vec->Evaluate(x).transpose();
    start += num;
    }*/

  return jac;
}
