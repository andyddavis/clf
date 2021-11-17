#include "clf/DenseQuadraticCostFunction.hpp"

using namespace clf;

DenseQuadraticCostFunction::DenseQuadraticCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions, std::size_t const outputDimension) : QuadraticCostFunction<Eigen::MatrixXd>(inputDimension, numPenaltyFunctions, outputDimension) {}

void DenseQuadraticCostFunction::Jacobian(Eigen::MatrixXd& jac) const {
  jac.resize(numPenaltyTerms, inputDimension);
  std::size_t count = 0;
  std::size_t cnt = 0;
  for( std::size_t i=0; i<numPenaltyFunctions; ++i ) {
    if( i>=outputDimension[count].first ) { ++count; }
    jac.block(cnt, 0, outputDimension[count].second, inputDimension) = PenaltyFunctionJacobian(i);
    cnt += outputDimension[count].second;
  }
}

std::vector<Eigen::MatrixXd> DenseQuadraticCostFunction::PenaltyFunctionHessianByFD(std::size_t const ind, Eigen::VectorXd const& beta, FDOrder const order, double const dbeta) const { return PenaltyFunctionHessianImpl(ind, beta); }

std::vector<Eigen::MatrixXd> DenseQuadraticCostFunction::PenaltyFunctionHessianImpl(std::size_t const ind, Eigen::VectorXd const& beta) const { 
  assert(beta.size()==inputDimension);
  assert(ind<numPenaltyFunctions);
  return std::vector<Eigen::MatrixXd>(PenaltyFunctionOutputDimension(ind), Eigen::MatrixXd::Zero(inputDimension, inputDimension));
}
