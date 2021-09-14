#include "clf/CoupledSupportPoint.hpp"

namespace pt = boost::property_tree;
using namespace clf;

CoupledSupportPoint::CoupledSupportPoint(Eigen::VectorXd const& x, std::shared_ptr<const Model> const& model, pt::ptree const& pt) :
SupportPoint(x, model, pt),
c0(pt.get<double>("CouplingMagnitudeScale", 1.0)),
c1(pt.get<double>("CouplingExponentialScale", 1.0))
{}

std::shared_ptr<CoupledSupportPoint> CoupledSupportPoint::Construct(Eigen::VectorXd const& x, std::shared_ptr<const Model> const& model, pt::ptree const& pt) {return SupportPoint::Construct<CoupledSupportPoint>(x, model, pt); }

double CoupledSupportPoint::CouplingFunction(std::size_t const neighInd) const {
  if( neighInd>=squaredNeighborDistances.size() ) { return 0.0; }

  // the support points are sorted so that the last entry is the farest
  const double r2 = squaredNeighborDistances[neighInd]/(*(squaredNeighborDistances.end()-1));

  return ( r2+1.0e-12>=1.0? 0.0 : c0*std::exp(c1*(1.0-1.0/(1.0-r2))) );
}
