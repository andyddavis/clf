#include "clf/LocalFunctions.hpp"

namespace pt = boost::property_tree;
using namespace clf;

LocalFunctions::LocalFunctions(std::shared_ptr<SupportPointCloud> const& cloud, pt::ptree const& pt) : cloud(cloud) {
  ComputeOptimalCoefficients();
}

double LocalFunctions::ComputeOptimalCoefficients() {
  if( IndependentSupportPoints() ) { return ComputeIndependentSupportPoints(); }
  return ComputeCoupledSupportPoints();
}

double LocalFunctions::ComputeCoupledSupportPoints() {
  assert(false);
  return cost;
}


double LocalFunctions::ComputeIndependentSupportPoints() {
  cost = 0.0;
  for( auto point=cloud->Begin(); point!=cloud->End(); ++point ) { cost += (*point)->MinimizeUncoupledCost(); }
  cost /= cloud->NumSupportPoints();
  return cost;
}

double LocalFunctions::CoefficientCost() const { return cost; }

Eigen::VectorXd LocalFunctions::Evaluate(Eigen::VectorXd const& x) const { return cloud->GetSupportPoint(NearestNeighborIndex(x))->EvaluateLocalFunction(x); }

std::size_t LocalFunctions::NearestNeighborIndex(Eigen::VectorXd const& x) const { return NearestNeighbor(x).first; }

double LocalFunctions::NearestNeighborDistance(Eigen::VectorXd const& x) const { return NearestNeighbor(x).second; }

std::pair<std::size_t, double> LocalFunctions::NearestNeighbor(Eigen::VectorXd const& x) const {
  // find the closest point to the input point
  std::vector<std::size_t> ind;
  std::vector<double> dist;
  cloud->FindNearestNeighbors(x, 1, ind, dist);
  return std::pair<std::size_t, double>(ind[0], dist[0]);
}

bool LocalFunctions::IndependentSupportPoints() const { return true; }
