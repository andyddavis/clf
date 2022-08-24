#include "clf/ConservationLawWeakFormResidual.hpp"

using namespace clf;

ConservationLawWeakFormResidual::ConservationLawWeakFormResidual(std::shared_ptr<LocalFunction> const& func, std::shared_ptr<ConservationLaw> const& system, std::shared_ptr<FeatureVector> const& testFunctionBasis, std::shared_ptr<const Parameters> const& para) :
  DensePenaltyFunction(func->NumCoefficients(), func->NumCoefficients(), para),
  function(func),
  system(system),
  testFunctionBasis(testFunctionBasis)
{
  GeneratePoints(para->Get<std::size_t>("NumPoints"));
}

void ConservationLawWeakFormResidual::GeneratePoints(std::size_t const num) {
  points = std::make_shared<PointCloud>(function->GetDomain());
  points->AddPoints(num);
  points->AddBoundaryPoints(num);
}

Eigen::VectorXd ConservationLawWeakFormResidual::Evaluate(Eigen::VectorXd const& beta) {
  Eigen::VectorXd eval = Eigen::VectorXd::Zero(outdim);

  for( std::size_t i=0; i<points->NumBoundaryPoints(); ++i ) {
    auto pt = points->GetBoundary(i);
    eval += testFunctionBasis->Evaluate(pt->x)*pt->normal->dot(system->Flux(function, pt->x, beta))/points->NumBoundaryPoints();
  }

  for( std::size_t i=0; i<points->NumPoints(); ++i ) {
    auto pt = points->Get(i);
    eval -= ( testFunctionBasis->Derivative(pt->x, Eigen::MatrixXi::Identity(testFunctionBasis->InputDimension(), testFunctionBasis->InputDimension()))*system->Flux(function, pt->x, beta) + testFunctionBasis->Evaluate(pt->x)*system->RightHandSide(pt->x) [0] )/points->NumPoints();
  }

  return eval;
}

Eigen::MatrixXd ConservationLawWeakFormResidual::Jacobian(Eigen::VectorXd const& beta) {
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(indim, outdim);

  for( std::size_t i=0; i<points->NumBoundaryPoints(); ++i ) {
    auto pt = points->GetBoundary(i);
    jac += testFunctionBasis->Evaluate(pt->x)*(pt->normal->transpose()*system->Flux_JacobianWRTCoefficients(function, pt->x, beta)/points->NumBoundaryPoints());
  }

  for( std::size_t i=0; i<points->NumPoints(); ++i ) {
    auto pt = points->Get(i);
    jac -= testFunctionBasis->Derivative(pt->x, Eigen::MatrixXi::Identity(testFunctionBasis->InputDimension(), testFunctionBasis->InputDimension()))*system->Flux_JacobianWRTCoefficients(function, pt->x, beta)/points->NumPoints();
  }

  return jac;
}

Eigen::MatrixXd ConservationLawWeakFormResidual::Hessian(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) {
  Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(outdim, outdim);

  for( std::size_t i=0; i<points->NumBoundaryPoints(); ++i ) {
    auto pt = points->GetBoundary(i);
    hess += (testFunctionBasis->Evaluate(pt->x).array()*weights.array()).sum()*system->Flux_HessianWRTCoefficients(function, pt->x, beta, *pt->normal)/points->NumBoundaryPoints();
  }

  for( std::size_t i=0; i<points->NumPoints(); ++i ) {
    auto pt = points->Get(i);
    const Eigen::MatrixXd phiDeriv = testFunctionBasis->Derivative(pt->x, Eigen::MatrixXi::Identity(testFunctionBasis->InputDimension(), testFunctionBasis->InputDimension())).array().colwise()*weights.array()/points->NumPoints();

    for( std::size_t j=0; j<phiDeriv.rows(); ++j ) { hess -= system->Flux_HessianWRTCoefficients(function, pt->x, beta, phiDeriv.row(j)); }
  }


  return hess;
}

std::size_t ConservationLawWeakFormResidual::NumPoints() const { return points->NumPoints(); }

std::shared_ptr<Point> ConservationLawWeakFormResidual::GetPoint(std::size_t const ind) const { return points->Get(ind); }

std::size_t ConservationLawWeakFormResidual::NumBoundaryPoints() const { return points->NumBoundaryPoints(); }

std::shared_ptr<Point> ConservationLawWeakFormResidual::GetBoundaryPoint(std::size_t const ind) const { return points->GetBoundary(ind); }
