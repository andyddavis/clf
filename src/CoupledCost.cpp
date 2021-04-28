#include "clf/CoupledCost.hpp"

#include "clf/SupportPoint.hpp"

namespace pt = boost::property_tree;
using namespace clf;

CoupledCost::CoupledCost(std::shared_ptr<SupportPoint> const& point, std::shared_ptr<SupportPoint> const& neighbor, pt::ptree const& pt) :
CostFunction(Eigen::VectorXi::Constant(1, point->NumCoefficients()+neighbor->NumCoefficients())),
coupledScale(pt.get<double>("CoupledScale", 1.0)),
point(point),
neighbor(neighbor),
localNeighborInd(LocalIndex(point, neighbor))
{}

bool CoupledCost::Coupled() const { return localNeighborInd!=std::numeric_limits<std::size_t>::max(); }

std::size_t CoupledCost::LocalIndex(std::shared_ptr<SupportPoint> const& point, std::shared_ptr<SupportPoint> const& neighbor) {
  const std::size_t localInd = point->LocalIndex(neighbor->GlobalIndex());
  return (localInd==0? std::numeric_limits<std::size_t>::max() : localInd);
}

double CoupledCost::CostImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& input) {
  if( !Coupled() ) { return 0.0; }

  auto pnt = point.lock();
  auto neigh = neighbor.lock();

  // coefficients for the point and the neighbor point
  const Eigen::Map<const Eigen::VectorXd> pointCoeffs(&input[0] (0), pnt->NumCoefficients());
  const Eigen::Map<const Eigen::VectorXd> neighCoeffs(&input[0] (pnt->NumCoefficients()), neigh->NumCoefficients());

  // the difference in the support point output (evaluated at the neighbor point)
  const Eigen::VectorXd diff = pnt->EvaluateLocalFunction(neigh->x, pointCoeffs) - neigh->EvaluateLocalFunction(neigh->x, neighCoeffs);

  return coupledScale*diff.dot(diff)/2.0;
}
