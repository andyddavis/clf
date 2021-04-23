#include "clf/LocalFunction.hpp"

namespace pt = boost::property_tree;
using namespace clf;

LocalFunction::LocalFunction(std::shared_ptr<SupportPointCloud> const& cloud, pt::ptree const& pt) : cloud(cloud) {
  ComputeOptimalCoefficients();
}

void LocalFunction::ComputeOptimalCoefficients() {
  IndependentSupportPoints();
}

void LocalFunction::IndependentSupportPoints() {
  cost = 0.0;
  for( auto point=cloud->Begin(); point!=cloud->End(); ++point ) {
    cost += (*point)->MinimizeUncoupledCost();
  }
  cost /= cloud->NumSupportPoints();
}

double LocalFunction::CoefficientCost() const { return cost; }

Eigen::VectorXd LocalFunction::Evaluate(Eigen::VectorXd const& x) const {
  // find the closest point to the input point
  std::vector<std::size_t> ind;
  std::vector<double> dist;
  cloud->FindNearestNeighbors(x, 1, ind, dist);

  return cloud->GetSupportPoint(ind[0])->EvaluateLocalFunction(x);
}
