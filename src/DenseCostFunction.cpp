#include "clf/DenseCostFunction.hpp"

using namespace clf;

DenseCostFunction::DenseCostFunction(DensePenaltyFunctions const& penaltyFunctions) : CostFunction<Eigen::MatrixXd>(penaltyFunctions) {}

DenseCostFunction::DenseCostFunction(std::shared_ptr<DensePenaltyFunction> const& penaltyFunction) : CostFunction<Eigen::MatrixXd>(penaltyFunction) {}

Eigen::MatrixXd DenseCostFunction::Jacobian(Eigen::VectorXd const& beta) const {
  assert(beta.size()==InputDimension());
  
  const std::size_t indim = InputDimension();
  Eigen::MatrixXd jac(numTerms, indim); 
  std::size_t start = 0;
  for( const auto& it : penaltyFunctions ) {
    assert(it);

    jac.block(start, 0, it->OutputDimension(), indim) = it->Jacobian(beta);
    start += it->OutputDimension();
  }
  assert(start==numTerms);

  return jac;
}

