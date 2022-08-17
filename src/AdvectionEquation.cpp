#include "clf/AdvectionEquation.hpp"

using namespace clf;

AdvectionEquation::AdvectionEquation(std::size_t const indim, double const constantVel, std::shared_ptr<const Parameters> const& para) :
  ConservationLaw(indim, para), constantVel(constantVel)
{}

AdvectionEquation::AdvectionEquation(Eigen::VectorXd const& vel, std::shared_ptr<const Parameters> const& para) :
  ConservationLaw(vel.size()+1, para), vel(vel)
{}

Eigen::VectorXd AdvectionEquation::Flux(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const {
  assert(u->InputDimension()==indim);
  assert(u->OutputDimension()==1);
  assert(u->NumCoefficients()==coeff.size());
  assert(x.size()==indim);

  Eigen::VectorXd output(indim);
  
  const double eval = u->Evaluate(x, coeff) [0];
  output(0) = eval;
  if( constantVel ) {
    output.tail(indim-1) = Eigen::VectorXd::Constant(indim-1, (*constantVel) * eval);
  } else {
    assert(vel);
    output.tail(indim-1) = (*vel) * eval;
  }
  return output;
}

double AdvectionEquation::FluxDivergence(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const {
  assert(u->InputDimension()==indim);
  assert(u->OutputDimension()==1);
  assert(u->NumCoefficients()==coeff.size());
  assert(x.size()==indim);

  const Eigen::VectorXd diff = u->Derivative(x, coeff, linOper).transpose();

  if( constantVel ) { return diff(0) + (*constantVel) * diff.tail(indim-1).sum(); }
  assert(vel);
  return diff(0) + (diff.tail(indim-1).array()*(*vel).array()).sum();
}

Eigen::VectorXd AdvectionEquation::FluxDivergence_GradientWRTCoefficients(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const {
  const std::optional<Eigen::VectorXd> y = u->featureMatrix->LocalCoordinate(x);
  const std::optional<Eigen::VectorXd>& jac = u->featureMatrix->LocalCoordinateJacobian();
  Eigen::MatrixXd phi = u->featureMatrix->Begin()->first->Derivative((y? *y : x), linOper->Counts(0).first, jac).transpose();

  if( constantVel ) {
    phi.block(1, 0, indim-1, phi.cols()) *= (*constantVel);
    return phi.colwise().sum();
  }
  
  assert(vel);
  phi.block(1, 0, indim-1, phi.cols()).array().colwise() *= vel->array();
  return phi.colwise().sum();
}

Eigen::MatrixXd AdvectionEquation::FluxDivergence_HessianWRTCoefficients(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const { return Eigen::MatrixXd::Zero(coeff.size(), coeff.size()); }
