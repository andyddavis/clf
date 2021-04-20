#include "clf/UncoupledCost.hpp"

#include "clf/SupportPoint.hpp"

using namespace muq::Optimization;
using namespace clf;

UncoupledCost::UncoupledCost(std::shared_ptr<SupportPoint> const& point) : CostFunction(Eigen::VectorXi::Constant(1, point->NumCoefficients())), point(point) {}

double UncoupledCost::CostImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& input) {
  std::cout << "HERE" << std::endl;
  return 0.0;
}
