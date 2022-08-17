#include "clf/BurgersEquation.hpp"

using namespace clf;

BurgersEquation::BurgersEquation(std::size_t const indim, double const constantVel, std::shared_ptr<const Parameters> const& para) :
  ConservationLaw(indim, para), constantVel(constantVel)
{}

BurgersEquation::BurgersEquation(Eigen::VectorXd const& vel, std::shared_ptr<const Parameters> const& para) :
  ConservationLaw(vel.size()+1, para), vel(vel)
{}

Eigen::VectorXd BurgersEquation::Flux(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const {
  assert(u->InputDimension()==indim);
  assert(u->OutputDimension()==1);
  assert(u->NumCoefficients()==coeff.size());
  assert(x.size()==indim);
  
  Eigen::VectorXd output(indim);
  
  const double eval = u->Evaluate(x, coeff) [0];
  output(0) = eval;
  if( constantVel ) {
    output.tail(indim-1) = Eigen::VectorXd::Constant(indim-1, (*constantVel) * 0.5*eval*eval);
  } else {
    assert(vel);
    output.tail(indim-1) = (*vel) * 0.5*eval*eval;
  }
  return output;
}

double BurgersEquation::FluxDivergence(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const {
  assert(u->InputDimension()==indim);
  assert(u->OutputDimension()==1);
  assert(u->NumCoefficients()==coeff.size());
  assert(x.size()==indim);

  const double eval = u->Evaluate(x, coeff) [0];
  const Eigen::VectorXd diff = u->Derivative(x, coeff, linOper).transpose();

  if( constantVel ) { return diff(0) + eval * (*constantVel) * diff.tail(indim-1).sum(); }
  assert(vel);
  return diff(0) + eval*(diff.tail(indim-1).array()*(*vel).array()).sum();
}

Eigen::VectorXd BurgersEquation::FluxDivergence_GradientWRTCoefficients(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const {
  const std::optional<Eigen::VectorXd> y = u->featureMatrix->LocalCoordinate(x);
  const std::optional<Eigen::VectorXd>& jac = u->featureMatrix->LocalCoordinateJacobian();
  const Eigen::VectorXd phi = u->featureMatrix->Begin()->first->Evaluate((y? *y : x));
  Eigen::MatrixXd phiDeriv = u->featureMatrix->Begin()->first->Derivative((y? *y : x), linOper->Counts(0).first, jac).transpose();
  
  const double eval = phi.dot(coeff);
  const Eigen::VectorXd diff = phiDeriv*coeff;

  if( constantVel ) {
    phiDeriv.block(1, 0, indim-1, phiDeriv.cols()) *= (*constantVel)*eval;
    return phiDeriv.colwise().sum().transpose() + (*constantVel)*diff.tail(indim-1).sum()*phi;
  } 

  assert(vel);
  phiDeriv.block(1, 0, indim-1, phiDeriv.cols()).array().colwise() *= vel->array()*eval;
  return phiDeriv.colwise().sum().transpose() + vel->dot(diff.tail(indim-1))*phi;
}

Eigen::MatrixXd BurgersEquation::FluxDivergence_HessianWRTCoefficients(std::shared_ptr<LocalFunction> const &u, Eigen::VectorXd const &x, Eigen::VectorXd const &coeff) const {
  const std::optional<Eigen::VectorXd> y = u->featureMatrix->LocalCoordinate(x);
  const std::optional<Eigen::VectorXd>& jac = u->featureMatrix->LocalCoordinateJacobian();
  const Eigen::VectorXd phi = u->featureMatrix->Begin()->first->Evaluate((y? *y : x));
  Eigen::MatrixXd phiDeriv = u->featureMatrix->Begin()->first->Derivative((y? *y : x), linOper->Counts(0).first, jac).transpose();

  if( constantVel ) {
    phiDeriv.block(1, 0, indim-1, phiDeriv.cols()) *= (*constantVel);
  } else {
    assert(vel);
    phiDeriv.block(1, 0, indim-1, phiDeriv.cols()).array().colwise() *= vel->array();
  }

  Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(coeff.size(), coeff.size());
  for( std::size_t i=1; i<indim; ++i ) { hess += phi*phiDeriv.row(i); }
  return hess + hess.transpose();
}
