#include "clf/SparseCostFunction.hpp"

using namespace clf;

SparseCostFunction::SparseCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions, std::size_t const outputDimension) : CostFunction(inputDimension, numPenaltyFunctions, outputDimension) {}

std::vector<Eigen::Triplet<double> > SparseCostFunction::PenaltyFunctionJacobianSparse(std::size_t const ind, Eigen::VectorXd const& beta) const {
  assert(beta.size()==inputDimension);
  assert(ind<numPenaltyFunctions);
  const std::vector<Eigen::Triplet<double> > jac = PenaltyFunctionJacobianSparseImpl(ind, beta);
  return jac;
}

void SparseCostFunction::Jacobian(Eigen::VectorXd const& beta, Eigen::SparseMatrix<double>& jac) const {
  // resize the jacobian---this sets every entry to zero, but does not free the memory
  jac.resize(numPenaltyTerms, inputDimension);
  std::vector<Eigen::Triplet<double> > triplets;
  std::size_t count = 0;
  std::size_t cnt = 0;
  for( std::size_t i=0; i<numPenaltyFunctions; ++i ) {
    if( i>=outputDimension[count].first ) { ++count; }
    const std::vector<Eigen::Triplet<double> > jaci = PenaltyFunctionJacobianSparse(i, beta);
    for( const auto& it : jaci ) { triplets.emplace_back(cnt+it.row(), it.col(), it.value()); }
    cnt += outputDimension[count].second;
  }
  jac.setFromTriplets(triplets.begin(), triplets.end());
  jac.makeCompressed();
}

Eigen::MatrixXd SparseCostFunction::PenaltyFunctionJacobianImpl(std::size_t const ind, Eigen::VectorXd const& beta) const {
  const std::vector<Eigen::Triplet<double> > sparseJac = PenaltyFunctionJacobianSparseImpl(ind, beta);

  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(PenaltyFunctionOutputDimension(ind), inputDimension);
  for( const auto& it : sparseJac ) { jac(it.row(), it.col()) = it.value(); }
  return jac;
}

std::vector<Eigen::Triplet<double> > SparseCostFunction::PenaltyFunctionJacobianSparseImpl(std::size_t const ind, Eigen::VectorXd const& beta) const {
  const std::size_t outputDimension = PenaltyFunctionOutputDimension(ind);
  const Eigen::MatrixXd jac = PenaltyFunctionJacobianByFD(ind, beta);
  std::vector<Eigen::Triplet<double> > sparseJac;
  for( std::size_t i=0; i<outputDimension; ++i ) {
    for( std::size_t j=0; j<inputDimension; ++j ) {
      if( std::abs(jac(i, j))>sparsityTol ) { sparseJac.emplace_back(i, j, jac(i, j)); }
    }
  }
  return sparseJac;
}

Eigen::SparseMatrix<double> SparseCostFunction::Hessian(Eigen::VectorXd const& beta, bool const gn) {
  assert(false);
}
