#include "clf/UncoupledCost.hpp"

#include "clf/SupportPoint.hpp"

namespace pt = boost::property_tree;
using namespace clf;

UncoupledCost::UncoupledCost(std::shared_ptr<SupportPoint> const& point, pt::ptree const& pt) :
DenseCostFunction(point->NumCoefficients(), point->NumNeighbors() + (pt.get<double>("RegularizationParameter", 0.0)<1.0e-14? 0.0 : 1)),
point(point),
uncoupledScale(std::sqrt(0.5*pt.get<double>("UncoupledScale", 1.0))),
regularizationScale(std::sqrt(0.5*pt.get<double>("RegularizationParameter", 0.0)))
{}

double UncoupledCost::UncoupledScale() const { return 2.0*uncoupledScale*uncoupledScale; }

double UncoupledCost::RegularizationScale() const { return 2.0*regularizationScale*regularizationScale; }

double UncoupledCost::PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& coefficients) const {
  // get the support point
  auto pnt = point.lock();
  assert(pnt);

  // the regularization cost term
  if( ind==pnt->NumNeighbors() ) { 
    assert(regularizationScale>1.0e-14);
    return regularizationScale*coefficients.norm();
  }

  // get the kernel evaluation 
  const double kernel = uncoupledScale*std::sqrt(pnt->NearestNeighborKernel(ind));

  // get the ind^th support point
  std::shared_ptr<SupportPoint> neigh = pnt->NearestNeighbor(ind);

  // compute the the residual evaluated at the neighbor's support point 
  return kernel*(neigh->model->Operator(neigh->x, coefficients, pnt->GetBasisFunctions()) - EvaluateForcingFunction(neigh)).norm();
}

Eigen::VectorXd UncoupledCost::PenaltyFunctionGradientImpl(std::size_t const ind, Eigen::VectorXd const& coefficients) const { 
  // get the support point
  auto pnt = point.lock();
  assert(pnt);

  // the regularization cost term
  if( ind==pnt->NumNeighbors() ) { 
    assert(regularizationScale>1.0e-14);
    const double coeffNorm = coefficients.norm();
    return (coeffNorm<1.0e-14? Eigen::VectorXd::Zero(inputDimension).eval() : (regularizationScale*coefficients/coeffNorm).eval());
  }
    
  // get the ind^th support point
  std::shared_ptr<SupportPoint> neigh = pnt->NearestNeighbor(ind);

  // the residual vector 
  const Eigen::VectorXd resid = neigh->model->Operator(neigh->x, coefficients, pnt->GetBasisFunctions()) - EvaluateForcingFunction(neigh);
  const double residNorm = resid.norm();

  // if the residual is zero, the gradient is also zero 
  if( residNorm<1.0e-14 ) { return Eigen::VectorXd::Zero(inputDimension); }

  // get the kernel evaluation 
  const double kernel = uncoupledScale*std::sqrt(pnt->NearestNeighborKernel(ind));

  // compute the gradient vector
  return kernel*neigh->model->OperatorJacobian(neigh->x, coefficients, pnt->GetBasisFunctions()).transpose()*resid/residNorm;
}

void UncoupledCost::SetForcingEvaluations(Eigen::MatrixXd const& force) { forcing = force; }

void UncoupledCost::UnsetForcingEvaluations() { forcing = boost::none; }

Eigen::VectorXd UncoupledCost::EvaluateForcingFunction(std::shared_ptr<SupportPoint> const& pnt) const {
  // if we have not set the precomputed forcing 
  if( forcing==boost::none ) { return pnt->model->RightHandSide(pnt->x); }

  // otherwise use the precomputed value
  return (*forcing).col(pnt->GlobalIndex());
}

bool UncoupledCost::IsQuadratic() const {
  // get the support point
  auto pnt = point.lock();
  assert(pnt);

  // if a model is nonlinear, than this cost function is not quadratic
  for( std::size_t i=0; i<pnt->NumNeighbors(); ++i ) { if( !pnt->NearestNeighbor(i)->model->IsLinear() ) { return false; } }

  return true;
}
