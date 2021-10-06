#include "clf/SparseQuadraticCostFunction.hpp"

using namespace clf;

SparseQuadraticCostFunction::SparseQuadraticCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions, std::size_t const outputDimension) : QuadraticCostFunction<Eigen::SparseMatrix<double> >(inputDimension, numPenaltyFunctions, outputDimension) {}

std::vector<std::pair<std::size_t, double> > SparseQuadraticCostFunction::PenaltyFunctionJacobianSparse(std::size_t const ind) const {
  assert(ind<numPenaltyFunctions);
  const std::vector<std::pair<std::size_t, double> > grad = PenaltyFunctionJacobianSparseImpl(ind);
  assert(grad.size()<=inputDimension);
  return grad;
}

void SparseQuadraticCostFunction::Jacobian(Eigen::SparseMatrix<double>& jac) const {
  // resize the jacobian---this sets every entry to zero, but does not free the memory
  jac.resize(numPenaltyFunctions, inputDimension);
  std::vector<Eigen::Triplet<double> > triplets;
  for( std::size_t i=0; i<numPenaltyFunctions; ++i ) {
    const std::vector<std::pair<std::size_t, double> > rowi = PenaltyFunctionJacobianSparse(i);
    for( const auto& it : rowi ) { triplets.emplace_back(i, it.first, it.second); }
  }
  jac.setFromTriplets(triplets.begin(), triplets.end());
  jac.makeCompressed();
}

std::vector<std::pair<std::size_t, double> > SparseQuadraticCostFunction::PenaltyFunctionJacobianSparseImpl(std::size_t const ind) const {
  const Eigen::VectorXd grad = PenaltyFunctionJacobianByFD(ind, Eigen::VectorXd::Zero(inputDimension));
  std::vector<std::pair<std::size_t, double> > sparseGrad;
  for( std::size_t i=0; i<inputDimension; ++i ) {
    if( std::abs(grad(i))>sparsityTol ) { sparseGrad.emplace_back(i, grad(i)); }
  }
  return sparseGrad;
}

Eigen::MatrixXd SparseQuadraticCostFunction::PenaltyFunctionJacobianImpl(std::size_t const ind) const {
  const std::vector<std::pair<std::size_t, double> > sparseGrad = PenaltyFunctionJacobianSparseImpl(ind);

  Eigen::VectorXd grad = Eigen::VectorXd::Zero(inputDimension);
  for( const auto& it : sparseGrad ) { grad(it.first) = it.second; }
  return grad;
}
