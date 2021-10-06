#include "clf/DenseCostFunction.hpp"

using namespace clf;

DenseCostFunction::DenseCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions, std::size_t const outputDimension) : CostFunction(inputDimension, numPenaltyFunctions, outputDimension) {}

DenseCostFunction::DenseCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions, std::vector<std::pair<std::size_t, std::size_t> > const& outputDimension) : CostFunction(inputDimension, numPenaltyFunctions, outputDimension) {}

void DenseCostFunction::Jacobian(Eigen::VectorXd const& beta, Eigen::MatrixXd& jac) const {
  jac.resize(numPenaltyTerms, inputDimension);
  std::size_t count = 0;
  std::size_t cnt = 0;
  for( std::size_t i=0; i<numPenaltyFunctions; ++i ) {
    if( i>=outputDimension[count].first ) { ++count; }
    jac.block(cnt, 0, outputDimension[count].second, inputDimension) = PenaltyFunctionJacobian(i, beta);
    cnt += outputDimension[count].second;
  }
}
