#include "clf/IdentityModel.hpp"

using namespace clf;

IdentityModel::IdentityModel(std::size_t const indim, std::size_t const outdim, std::shared_ptr<const Parameters> const& para) :
  SystemOfEquations(indim, outdim, para)
{}

IdentityModel::IdentityModel(std::shared_ptr<const Parameters> const& para) :
  SystemOfEquations(para) 
{}

Eigen::VectorXd IdentityModel::Operator(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const { 
  assert(u->InputDimension()==indim);
  assert(u->OutputDimension()==outdim);
  assert(u->NumCoefficients()==coeff.size());
  assert(x.size()==indim);
  return u->Evaluate(x, coeff);
}

Eigen::MatrixXd IdentityModel::JacobianWRTCoefficients(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const {
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(outdim, coeff.size());

  std::size_t start = 0;
  std::size_t row = 0;
  const std::optional<Eigen::VectorXd> y = u->featureMatrix->LocalCoordinate(x);
  for( auto it=u->featureMatrix->Begin(); it!=u->featureMatrix->End(); ++it ) {
    const Eigen::VectorXd phi = it->first->Evaluate((y? *y : x));
    for( std::size_t i=0; i<it->second; ++i ) {
      jac.row(row++).segment(start, phi.size()) = phi;
      start += phi.size();
    }
  }

  return jac;
}

Eigen::MatrixXd IdentityModel::HessianWRTCoefficients(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff, Eigen::VectorXd const& weights) const { return Eigen::MatrixXd::Zero(coeff.size(), coeff.size()); }
