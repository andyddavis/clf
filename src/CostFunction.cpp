#include "clf/CostFunction.hpp"

using namespace clf;

DenseCostFunction::DenseCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions) : CostFunction(inputDimension, numPenaltyFunctions) {}

void DenseCostFunction::Jacobian(Eigen::VectorXd const& beta, Eigen::MatrixXd& jac) const {
  jac.resize(numPenaltyFunctions, inputDimension);
  for( std::size_t i=0; i<numPenaltyFunctions; ++i ) { jac.row(i) = PenaltyFunctionGradient(i, beta); }
}

SparseCostFunction::SparseCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions) : CostFunction(inputDimension, numPenaltyFunctions) {}

std::vector<std::pair<std::size_t, double> > SparseCostFunction::PenaltyFunctionGradientSparse(std::size_t const ind, Eigen::VectorXd const& beta) const {
  assert(beta.size()==inputDimension);
  assert(ind<numPenaltyFunctions);
  const std::vector<std::pair<std::size_t, double> > grad = PenaltyFunctionGradientSparseImpl(ind, beta);
  assert(grad.size()<=inputDimension);
  return grad;
}

void SparseCostFunction::Jacobian(Eigen::VectorXd const& beta, Eigen::SparseMatrix<double>& jac) const {
  // resize the jacobian---this sets every entry to zero, but does not free the memory
  jac.resize(numPenaltyFunctions, inputDimension);
  std::vector<Eigen::Triplet<double> > triplets;
  for( std::size_t i=0; i<numPenaltyFunctions; ++i ) {
    const std::vector<std::pair<std::size_t, double> > rowi = PenaltyFunctionGradientSparse(i, beta);
    for( const auto& it : rowi ) { triplets.emplace_back(i, it.first, it.second); }
  }
  jac.setFromTriplets(triplets.begin(), triplets.end());
  jac.makeCompressed();
}

Eigen::VectorXd SparseCostFunction::PenaltyFunctionGradientImpl(std::size_t const ind, Eigen::VectorXd const& beta) const {
  const std::vector<std::pair<std::size_t, double> > sparseGrad = PenaltyFunctionGradientSparseImpl(ind, beta);
  
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(inputDimension);
  for( const auto& it : sparseGrad ) { grad(it.first) = it.second; }
  return grad;
}

std::vector<std::pair<std::size_t, double> > SparseCostFunction::PenaltyFunctionGradientSparseImpl(std::size_t const ind, Eigen::VectorXd const& beta) const {
  const Eigen::VectorXd grad = PenaltyFunctionGradientByFD(ind, beta);
  std::vector<std::pair<std::size_t, double> > sparseGrad;
  for( std::size_t i=0; i<inputDimension; ++i ) {
    if( std::abs(grad(i))>sparsityTol ) { sparseGrad.emplace_back(i, grad(i)); }
  }
  return sparseGrad;
}
