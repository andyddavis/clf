#include "clf/SparseQuadraticCostFunction.hpp"

using namespace clf;

SparseQuadraticCostFunction::SparseQuadraticCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions, std::size_t const outputDimension) : QuadraticCostFunction<Eigen::SparseMatrix<double> >(inputDimension, numPenaltyFunctions, outputDimension) {}

std::vector<Eigen::Triplet<double> > SparseQuadraticCostFunction::PenaltyFunctionJacobianSparse(std::size_t const ind) const {
  assert(ind<numPenaltyFunctions);
  return PenaltyFunctionJacobianSparseImpl(ind);
}

void SparseQuadraticCostFunction::Jacobian(Eigen::SparseMatrix<double>& jac) const {
  // resize the jacobian---this sets every entry to zero, but does not free the memory
  jac.resize(numPenaltyTerms, inputDimension);
  std::vector<Eigen::Triplet<double> > triplets;
  std::size_t count = 0;
  std::size_t cnt = 0;
  for( std::size_t i=0; i<numPenaltyFunctions; ++i ) {
    if( i>=outputDimension[count].first ) { ++count; }
    const std::vector<Eigen::Triplet<double> > jaci = PenaltyFunctionJacobianSparse(i);
    for( const auto& it : jaci ) { triplets.emplace_back(cnt+it.row(), it.col(), it.value()); }
    cnt += outputDimension[count].second;
  }
  jac.setFromTriplets(triplets.begin(), triplets.end());
  jac.makeCompressed();
}

Eigen::VectorXd SparseQuadraticCostFunction::ApplyPenaltyFunctionJacobian(std::size_t const ind, Eigen::VectorXd const& beta) const {
  const std::vector<Eigen::Triplet<double> > triplets = PenaltyFunctionJacobianSparse(ind);

  Eigen::SparseMatrix<double> jac(PenaltyFunctionOutputDimension(ind), inputDimension);
  jac.setFromTriplets(triplets.begin(), triplets.end());
  return jac*beta;
}

Eigen::MatrixXd SparseQuadraticCostFunction::PenaltyFunctionJacobianImpl(std::size_t const ind) const {
  const std::vector<Eigen::Triplet<double> > sparseJac = PenaltyFunctionJacobianSparseImpl(ind);

  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(PenaltyFunctionOutputDimension(ind), inputDimension);
  for( const auto& it : sparseJac ) { jac(it.row(), it.col()) = it.value(); }
  return jac;
}

std::vector<Eigen::SparseMatrix<double> > SparseQuadraticCostFunction::PenaltyFunctionHessianByFD(std::size_t const ind, Eigen::VectorXd const& beta, FDOrder const order, double const dbeta) const { return PenaltyFunctionHessianImpl(ind, beta); }

std::vector<Eigen::SparseMatrix<double> > SparseQuadraticCostFunction::PenaltyFunctionHessianImpl(std::size_t const ind, Eigen::VectorXd const& beta) const {
  assert(beta.size()==inputDimension);
  assert(ind<numPenaltyFunctions);
  return std::vector<Eigen::SparseMatrix<double> >(PenaltyFunctionOutputDimension(ind), Eigen::SparseMatrix<double>(inputDimension, inputDimension));
}
