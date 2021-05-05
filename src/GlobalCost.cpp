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

    // if necessary, evaluate the coupled cost
    for( const auto& coupled : (*point)->coupledCost ) {
      assert(coupled);
      auto neigh = coupled->neighbor.lock();
      assert(neigh);

      // extract the coefficients associated with the neighbor
      Eigen::Map<const Eigen::VectorXd> coeffNeigh(&input[0](dofIndices.globalDoFIndices[neigh->GlobalIndex()]), neigh->NumCoefficients());
      cost += coupled->Cost(coeff, coeffNeigh);
    }
  }

  return cost;
}

void GlobalCost::GradientImpl(unsigned int const inputDimWrt, muq::Modeling::ref_vector<Eigen::VectorXd> const& input, Eigen::VectorXd const& sensitivity) {
  this->gradient = Eigen::VectorXd::Zero(inputSizes(0));
  assert(this->gradient.size()==input[0].get().size());

  // loop through the support points
  for( auto point=dofIndices.cloud->Begin(); point!=dofIndices.cloud->End(); ++point ) {
    const std::size_t globalDoFStart = dofIndices.globalDoFIndices[(*point)->GlobalIndex()];
    const std::size_t dofLength = (*point)->NumCoefficients();

    // extract the coefficients associated with (*point)
    Eigen::Map<const Eigen::VectorXd> coeff(&input[0](globalDoFStart), dofLength);

    // uncoupled gradient
    this->gradient.segment(globalDoFStart, dofLength) += (*point)->uncoupledCost->Gradient(coeff);

    // if necessary, evaluate the coupled cost
    for( const auto& coupled : (*point)->coupledCost ) {
      assert(coupled);
      auto neigh = coupled->neighbor.lock();
      assert(neigh);

      const std::size_t globalDoFStartNeigh = dofIndices.globalDoFIndices[neigh->GlobalIndex()];
      const std::size_t dofLengthNeigh = neigh->NumCoefficients();

      // extract the coefficients associated with the neighbor
      Eigen::Map<const Eigen::VectorXd> coeffNeigh(&input[0](globalDoFStartNeigh), dofLengthNeigh);

      const Eigen::VectorXd coupledGrad = coupled->Gradient(coeff, coeffNeigh);
      this->gradient.segment(globalDoFStart, dofLength) += coupledGrad.head(dofLength);
      this->gradient.segment(globalDoFStartNeigh, dofLengthNeigh) += coupledGrad.tail(dofLengthNeigh);
    }
  }

  this->gradient *= sensitivity(0);
}
