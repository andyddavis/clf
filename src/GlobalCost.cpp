#include "clf/GlobalCost.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::Optimization;
using namespace clf;

GlobalCost::GlobalCost(std::shared_ptr<SupportPointCloud> const& cloud, pt::ptree const& pt) :
CostFunction(Eigen::VectorXi::Constant(1, cloud->numCoefficients)),
dofIndices(cloud)
{}

double GlobalCost::CostImpl(ref_vector<Eigen::VectorXd> const& input) {
  assert(dofIndices.cloud);
  double cost = 0.0;

  // loop through the support points
  for( auto point=dofIndices.cloud->Begin(); point!=dofIndices.cloud->End(); ++point ) {
    // extract the coefficients associated with (*point)
    Eigen::Map<const Eigen::VectorXd> coeff(&input[0](dofIndices.globalDoFIndices[(*point)->GlobalIndex()]), (*point)->NumCoefficients());

    // evaluate the uncoupled cost function
    cost += (*point)->uncoupledCost->Cost(coeff);
  }

  return cost;
}
