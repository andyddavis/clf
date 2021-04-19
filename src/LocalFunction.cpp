#include "clf/LocalFunction.hpp"

namespace pt = boost::property_tree;
using namespace clf;

LocalFunction::LocalFunction(std::shared_ptr<SupportPointCloud> const& cloud, pt::ptree const& pt) : cloud(cloud) {
  ComputeOptimalCoefficients();
}

void LocalFunction::ComputeOptimalCoefficients() const {
  IndependentSupportPoints();
}

void LocalFunction::IndependentSupportPoints() const {
  for( auto point=cloud->Begin(); point!=cloud->End(); ++point ) { (*point)->MinimizeUncoupledCost(); }
}
