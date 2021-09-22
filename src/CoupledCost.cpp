#include "clf/CoupledCost.hpp"

#include "clf/SupportPoint.hpp"

namespace pt = boost::property_tree;
using namespace clf;

CoupledCost::CoupledCost(std::shared_ptr<SupportPoint> const& point, std::shared_ptr<SupportPoint> const& neighbor, pt::ptree const& pt) :
SparseCostFunction(point->NumCoefficients()+neighbor->NumCoefficients(), point->model->outputDimension),
point(point),
neighbor(neighbor),
pointBasisEvals(point->EvaluateBasisFunctions(neighbor->x)),
neighborBasisEvals(neighbor->EvaluateBasisFunctions(neighbor->x)),
localNeighborInd(LocalIndex(point, neighbor)),
scale((localNeighborInd==std::numeric_limits<std::size_t>::max()? 0 : std::sqrt(pt.get<double>("CoupledScale")*point->NearestNeighborKernel(localNeighborInd))))
{
  assert(pointBasisEvals.size()==numPenaltyFunctions);
  assert(neighborBasisEvals.size()==numPenaltyFunctions);
}

std::size_t CoupledCost::LocalIndex(std::shared_ptr<SupportPoint> const& point, std::shared_ptr<SupportPoint> const& neighbor) {
  const std::size_t localInd = point->LocalIndex(neighbor->GlobalIndex());
  return (localInd==0? std::numeric_limits<std::size_t>::max() : localInd);
}

bool CoupledCost::Coupled() const { return localNeighborInd!=std::numeric_limits<std::size_t>::max(); }

Eigen::VectorXd CoupledCost::ComputeCost(Eigen::VectorXd const& coeffPoint, Eigen::VectorXd const& coeffNeigh) const {
  if( !Coupled() ) { return Eigen::VectorXd::Zero(numPenaltyFunctions); }

  auto pnt = point.lock(); assert(pnt);
  auto neigh = neighbor.lock(); assert(neigh);

  // the difference in the support point output (evaluated at the neighbor point)
  return scale*(pnt->EvaluateLocalFunction(coeffPoint, pointBasisEvals) - neigh->EvaluateLocalFunction(coeffNeigh, neighborBasisEvals));
}

double CoupledCost::PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const {
  assert(false);
  return 0.0;
}

Eigen::VectorXd CoupledCost::CostImpl(Eigen::VectorXd const& beta) const {
  if( !Coupled() ) { return Eigen::VectorXd::Zero(numPenaltyFunctions); }

  auto pnt = point.lock(); assert(pnt);
  auto neigh = neighbor.lock(); assert(neigh);

  // the coefficients for the point and its neighbor
  const Eigen::Map<const Eigen::VectorXd> pointCoeffs(&beta(0), pnt->NumCoefficients());
  const Eigen::Map<const Eigen::VectorXd> neighCoeffs(&beta(pnt->NumCoefficients()), neigh->NumCoefficients());

  // the difference in the support point output (evaluated at the neighbor point)
  return ComputeCost(pointCoeffs, neighCoeffs);
}

void CoupledCost::JacobianTriplets(std::vector<Eigen::Triplet<double> >& triplets) const {
  auto pnt = point.lock(); assert(pnt);

  for( std::size_t i=0; i<numPenaltyFunctions; ++i ) {
    const std::size_t ind0 = (i==0? 0 : pointBasisEvals[i-1].size());
    const std::size_t ind1 = pnt->NumCoefficients() + (i==0? 0 : neighborBasisEvals[i-1].size());
    assert(pointBasisEvals[i].size()==neighborBasisEvals[i].size());

    for( std::size_t j=0; j<pointBasisEvals[i].size(); ++j ) {
      if( std::abs(pointBasisEvals[i][j])>1.0e-14 ) { triplets.emplace_back(i, ind0 + j, scale*pointBasisEvals[i][j]); }
      if( std::abs(neighborBasisEvals[i][j])>1.0e-14 ) { triplets.emplace_back(i, ind1 + j, -scale*neighborBasisEvals[i][j]); }
    }
  }
}

void CoupledCost::JacobianImpl(Eigen::VectorXd const& beta, Eigen::SparseMatrix<double>& jac) const {
  if( !Coupled() ) { return; }

  std::vector<Eigen::Triplet<double> > triplets;
  triplets.reserve(beta.size());
  JacobianTriplets(triplets);
  jac.setFromTriplets(triplets.begin(), triplets.end());
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
