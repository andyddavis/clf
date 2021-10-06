#include "clf/CoupledCost.hpp"

#include "clf/SupportPoint.hpp"

namespace pt = boost::property_tree;
using namespace clf;

CoupledCost::CoupledCost(std::shared_ptr<SupportPoint> const& point, std::shared_ptr<SupportPoint> const& neighbor, pt::ptree const& pt) :
DenseCostFunction(point->NumCoefficients()+neighbor->NumCoefficients(), 1, point->model->outputDimension),
point(point),
neighbor(neighbor),
pointBasisEvals(point->EvaluateBasisFunctions(neighbor->x)),
neighborBasisEvals(neighbor->EvaluateBasisFunctions(neighbor->x)),
localNeighborInd(LocalIndex(point, neighbor)),
scale((localNeighborInd==std::numeric_limits<std::size_t>::max()? 0.0 : std::sqrt(0.5*pt.get<double>("CoupledScale")*point->NearestNeighborKernel(localNeighborInd))))
{
  assert(pointBasisEvals.size()==point->model->outputDimension);
  assert(neighborBasisEvals.size()==point->model->outputDimension);
}

std::size_t CoupledCost::LocalIndex(std::shared_ptr<SupportPoint> const& point, std::shared_ptr<SupportPoint> const& neighbor) {
  const std::size_t localInd = point->LocalIndex(neighbor->GlobalIndex());
  return (localInd==0? std::numeric_limits<std::size_t>::max() : localInd);
}

bool CoupledCost::Coupled() const { return localNeighborInd!=std::numeric_limits<std::size_t>::max(); }

Eigen::VectorXd CoupledCost::PenaltyFunction(Eigen::VectorXd const& coeffPoint, Eigen::VectorXd const& coeffNeigh) const {
  if( !Coupled() ) {
    assert(false);
    return Eigen::VectorXd();
  }

  auto pnt = point.lock(); assert(pnt);
  auto neigh = neighbor.lock(); assert(neigh);

  // the difference in the support point output (evaluated at the neighbor point)
  return scale*(pnt->EvaluateLocalFunction(coeffPoint, pointBasisEvals) - neigh->EvaluateLocalFunction(coeffNeigh, neighborBasisEvals));
}

Eigen::VectorXd CoupledCost::PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const {
  if( !Coupled() ) {
    assert(false);
    return Eigen::VectorXd();
  }

  auto pnt = point.lock(); assert(pnt);
  auto neigh = neighbor.lock(); assert(neigh);

  // the coefficients for the point and its neighbor
  const Eigen::Map<const Eigen::VectorXd> pointCoeffs(&beta(0), pnt->NumCoefficients());
  const Eigen::Map<const Eigen::VectorXd> neighCoeffs(&beta(pnt->NumCoefficients()), neigh->NumCoefficients());

  return PenaltyFunction(pointCoeffs, neighCoeffs);
}

Eigen::MatrixXd CoupledCost::PenaltyFunctionJacobian(Eigen::VectorXd const& coeffPoint, Eigen::VectorXd const& coeffNeigh) const {
  if( !Coupled() ) { return Eigen::VectorXd::Zero(inputDimension); }

  auto pnt = point.lock(); assert(pnt);
  auto neigh = neighbor.lock(); assert(neigh);

  Eigen::VectorXd resid = pnt->EvaluateLocalFunction(coeffPoint, pointBasisEvals) - neigh->EvaluateLocalFunction(coeffNeigh, neighborBasisEvals);
  const double residNorm = resid.norm();
  if( residNorm<1.0e-14 ) { return Eigen::VectorXd::Zero(inputDimension); }
  assert(resid.size()==pointBasisEvals.size());
  assert(resid.size()==neighborBasisEvals.size());

  // scale the residual rather than the gradient because the output size will likely be much smaller than the number of inputs
  resid *= scale/residNorm;

  Eigen::VectorXd grad(inputDimension);
  std::size_t ind = 0;
  std::size_t jnd = pnt->NumCoefficients();
  for( std::size_t i=0; i<resid.size(); ++i ) {
    grad.segment(ind, pointBasisEvals[i].size()) = resid(i)*pointBasisEvals[i];
    ind += pointBasisEvals[i].size();
    grad.segment(jnd, neighborBasisEvals[i].size()) = -resid(i)*neighborBasisEvals[i];
    jnd += neighborBasisEvals[i].size();
  }

  return grad;
}

Eigen::MatrixXd CoupledCost::PenaltyFunctionJacobianImpl(std::size_t const ind, Eigen::VectorXd const& beta) const {
  if( !Coupled() ) { return Eigen::VectorXd::Zero(inputDimension); }

  auto pnt = point.lock(); assert(pnt);
  auto neigh = neighbor.lock(); assert(neigh);

  // the coefficients for the point and its neighbor
  const Eigen::Map<const Eigen::VectorXd> pointCoeffs(&beta(0), pnt->NumCoefficients());
  const Eigen::Map<const Eigen::VectorXd> neighCoeffs(&beta(pnt->NumCoefficients()), neigh->NumCoefficients());

  return PenaltyFunctionJacobian(pointCoeffs, neighCoeffs);
}

std::shared_ptr<const SupportPoint> CoupledCost::GetPoint() const {
  auto pnt = point.lock();
  assert(pnt);
  return pnt;
}

std::shared_ptr<const SupportPoint> CoupledCost::GetNeighbor() const {
  auto neigh = neighbor.lock();
  assert(neigh);
  return neigh;
}

double CoupledCost::CoupledScale() const {
  auto pnt = point.lock();
  assert(pnt);
  return (Coupled()? 2.0*scale*scale/pnt->NearestNeighborKernel(localNeighborInd) : 0.0);
}

bool CoupledCost::IsQuadratic() const { return true; }
