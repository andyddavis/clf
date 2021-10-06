#include "clf/DenseQuadraticCostFunction.hpp"

using namespace clf;

DenseQuadraticCostFunction::DenseQuadraticCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions, std::size_t const outputDimension) : QuadraticCostFunction<Eigen::MatrixXd>(inputDimension, numPenaltyFunctions, outputDimension) {}

void DenseQuadraticCostFunction::Jacobian(Eigen::MatrixXd& jac) const {
  jac.resize(numPenaltyFunctions, inputDimension);
  for( std::size_t i=0; i<numPenaltyFunctions; ++i ) { jac.row(i) = PenaltyFunctionGradient(i); }
}
