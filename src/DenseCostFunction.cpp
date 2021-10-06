#include "clf/DenseCostFunction.hpp"

using namespace clf;

DenseCostFunction::DenseCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions, std::size_t const outputDimension) : CostFunction(inputDimension, numPenaltyFunctions, outputDimension) {}

void DenseCostFunction::Jacobian(Eigen::VectorXd const& beta, Eigen::MatrixXd& jac) const {
  jac.resize(numPenaltyFunctions, inputDimension);
  for( std::size_t i=0; i<numPenaltyFunctions; ++i ) { jac.row(i) = PenaltyFunctionGradient(i, beta); }
}
