#include "clf/CoupledCost.hpp"

#include "clf/SupportPoint.hpp"

namespace pt = boost::property_tree;
using namespace clf;

CoupledCost::CoupledCost(std::shared_ptr<SupportPoint> const& point, std::shared_ptr<SupportPoint> const& neighbor, pt::ptree const& pt) :
SparseQuadraticCostFunction(point->NumCoefficients()+neighbor->NumCoefficients(), 1, point->model->outputDimension),
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
  if( !Coupled() ) { return Eigen::VectorXd(outputDimension[0].second); }

  auto pnt = point.lock(); assert(pnt);
  auto neigh = neighbor.lock(); assert(neigh);

  // the difference in the support point output (evaluated at the neighbor point)
  return scale*(pnt->EvaluateLocalFunction(coeffPoint, pointBasisEvals) - neigh->EvaluateLocalFunction(coeffNeigh, neighborBasisEvals));
}

std::vector<Eigen::Triplet<double> > CoupledCost::PenaltyFunctionJacobian() const {
  auto pnt = point.lock(); assert(pnt);
  auto neigh = neighbor.lock(); assert(neigh);

  if( !Coupled() ) { return std::vector<Eigen::Triplet<double> >(); }

  std::vector<Eigen::Triplet<double> > triplets; 

  std::size_t cnt = 0;
  for( std::size_t i=0; i<pnt->model->outputDimension; ++i ) {
    for( std::size_t j=0; j<pointBasisEvals[i].size(); ++j ) { 
      if( std::abs(pointBasisEvals[i](j))>1.0e-15 ) { triplets.emplace_back(i, cnt+j, scale*pointBasisEvals[i](j)); }		      
    }
    cnt += pointBasisEvals[i].size();
  }

  cnt = pnt->NumCoefficients();
  for( std::size_t i=0; i<neigh->model->outputDimension; ++i ) {
    for( std::size_t j=0; j<neighborBasisEvals[i].size(); ++j ) { 
      if( std::abs(neighborBasisEvals[i](j))>1.0e-15 ) { triplets.emplace_back(i, cnt+j, -scale*neighborBasisEvals[i](j)); }		      
    }
    cnt += neighborBasisEvals[i].size();
  }

  return triplets;
}

std::vector<Eigen::Triplet<double> > CoupledCost::PenaltyFunctionJacobianSparseImpl(std::size_t const ind) const { return PenaltyFunctionJacobian(); }

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

