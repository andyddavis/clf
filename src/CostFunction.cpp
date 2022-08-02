#include "clf/CostFunction.hpp"

using namespace clf;

DenseCostFunction::DenseCostFunction(DensePenaltyFunctions const& penaltyFunctions) : CostFunction<Eigen::MatrixXd>(penaltyFunctions) {}

Eigen::MatrixXd DenseCostFunction::Jacobian(Eigen::VectorXd const& beta) const {
  assert(beta.size()==InputDimension());
  
  const std::size_t indim = InputDimension();
  Eigen::MatrixXd jac(numTerms, indim); 
  std::size_t start = 0;
  for( const auto& it : penaltyFunctions ) {
    assert(it);

    jac.block(start, 0, it->outdim, indim) = it->Jacobian(beta);
    start += it->outdim;
  }
  assert(start==numTerms);

  return jac;
}

SparseCostFunction::SparseCostFunction(SparsePenaltyFunctions const& penaltyFunctions) : CostFunction<Eigen::SparseMatrix<double> >(penaltyFunctions) {}

Eigen::SparseMatrix<double> SparseCostFunction::Jacobian(Eigen::VectorXd const& beta) const { 
  assert(beta.size()==InputDimension());
  
  std::vector<Eigen::Triplet<double> > entries;
  std::size_t start = 0;
  for( const auto& it : penaltyFunctions ) {
    auto ptr = std::dynamic_pointer_cast<SparsePenaltyFunction>(it);
    assert(ptr);
    std::vector<Eigen::Triplet<double> > entriesLocal = ptr->JacobianEntries(beta);
    for( auto& jt : entriesLocal ) { entries.emplace_back(start+jt.row(), jt.col(), jt.value()); }
    start += it->outdim;
  }
  
  Eigen::SparseMatrix<double> jac(numTerms, InputDimension()); 
  jac.setFromTriplets(entries.begin(), entries.end());
  return jac;
}

