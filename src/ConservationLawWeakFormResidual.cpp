#include "clf/ConservationLawWeakFormResidual.hpp"

using namespace clf;

ConservationLawWeakFormResidual::ConservationLawWeakFormResidual(std::shared_ptr<LocalFunction> const& func, std::shared_ptr<ConservationLaw> const& system, std::shared_ptr<FeatureVector> const& testFunctionBasis, std::shared_ptr<const Parameters> const& para) :
  Residual(func, system, para),
  testFunctionBasis(testFunctionBasis)
{
  GeneratePoints(para->Get<std::size_t>("NumPoints"));
}

void ConservationLawWeakFormResidual::GeneratePoints(std::size_t const num) {
  cloud = std::make_shared<PointCloud>(function->GetDomain());
  cloud->AddPoints(num);
  cloud->AddBoundaryPoints(num);
}

Eigen::VectorXd ConservationLawWeakFormResidual::Evaluate(Eigen::VectorXd const& beta) {
  Eigen::VectorXd eval = Eigen::VectorXd::Zero(outdim);

  auto sys = std::dynamic_pointer_cast<const ConservationLaw>(system);
  assert(sys);

  for( std::size_t i=0; i<cloud->NumBoundaryPoints(); ++i ) {
    auto pt = cloud->GetBoundary(i);
    eval += testFunctionBasis->Evaluate(pt->x)*pt->normal->dot(sys->Flux(function, pt->x, beta))/cloud->NumBoundaryPoints();
  }

  for( std::size_t i=0; i<cloud->NumPoints(); ++i ) {
    auto pt = cloud->Get(i);
    eval -= ( testFunctionBasis->Derivative(pt->x, Eigen::MatrixXi::Identity(testFunctionBasis->InputDimension(), testFunctionBasis->InputDimension()))*sys->Flux(function, pt->x, beta) + testFunctionBasis->Evaluate(pt->x)*system->RightHandSide(pt->x) [0] )/cloud->NumPoints();
  }

  return eval;
}

Eigen::MatrixXd ConservationLawWeakFormResidual::Jacobian(Eigen::VectorXd const& beta) {
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(indim, outdim);

  auto sys = std::dynamic_pointer_cast<const ConservationLaw>(system);
  assert(sys);

  for( std::size_t i=0; i<cloud->NumBoundaryPoints(); ++i ) {
    auto pt = cloud->GetBoundary(i);
    jac += testFunctionBasis->Evaluate(pt->x)*(pt->normal->transpose()*sys->Flux_JacobianWRTCoefficients(function, pt->x, beta)/cloud->NumBoundaryPoints());
  }

  for( std::size_t i=0; i<cloud->NumPoints(); ++i ) {
    auto pt = cloud->Get(i);
    jac -= testFunctionBasis->Derivative(pt->x, Eigen::MatrixXi::Identity(testFunctionBasis->InputDimension(), testFunctionBasis->InputDimension()))*sys->Flux_JacobianWRTCoefficients(function, pt->x, beta)/cloud->NumPoints();
  }

  return jac;
}

Eigen::MatrixXd ConservationLawWeakFormResidual::Hessian(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) {
  Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(outdim, outdim);

  auto sys = std::dynamic_pointer_cast<const ConservationLaw>(system);
  assert(sys);

  for( std::size_t i=0; i<cloud->NumBoundaryPoints(); ++i ) {
    auto pt = cloud->GetBoundary(i);
    hess += (testFunctionBasis->Evaluate(pt->x).array()*weights.array()).sum()*sys->Flux_HessianWRTCoefficients(function, pt->x, beta, *pt->normal)/cloud->NumBoundaryPoints();
  }

  for( std::size_t i=0; i<cloud->NumPoints(); ++i ) {
    auto pt = cloud->Get(i);
    const Eigen::MatrixXd phiDeriv = testFunctionBasis->Derivative(pt->x, Eigen::MatrixXi::Identity(testFunctionBasis->InputDimension(), testFunctionBasis->InputDimension())).array().colwise()*weights.array()/cloud->NumPoints();

    for( std::size_t j=0; j<phiDeriv.rows(); ++j ) { hess -= sys->Flux_HessianWRTCoefficients(function, pt->x, beta, phiDeriv.row(j)); }
  }

  return hess;
}
