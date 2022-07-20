#include "clf/CostFunction.hpp"

using namespace clf;

DenseCostFunction::DenseCostFunction(DensePenaltyFunctions const& penaltyFunctions) : CostFunction<Eigen::MatrixXd>(penaltyFunctions) {}

void DenseCostFunction::Jacobian(Eigen::VectorXd const& beta, Eigen::MatrixXd& jac) const {
  assert(beta.size()==InputDimension());
  
  const std::size_t indim = InputDimension();
  jac.resize(numTerms, indim); 
  std::size_t start = 0;
  for( const auto& it : penaltyFunctions ) {
    assert(it);

    jac.block(start, 0, it->outdim, indim) = it->Jacobian(beta);
    start += it->outdim;
  }
  assert(start==numTerms);
}

SparseCostFunction::SparseCostFunction(SparsePenaltyFunctions const& penaltyFunctions) : CostFunction<Eigen::SparseMatrix<double> >(penaltyFunctions) {}

void SparseCostFunction::Jacobian(Eigen::VectorXd const& beta, Eigen::SparseMatrix<double>& jac) const { 
  assert(beta.size()==InputDimension());
  
  std::vector<Eigen::Triplet<double> > entries;
  std::size_t start = 0;
  for( const auto& it : penaltyFunctions ) {
    std::vector<Eigen::Triplet<double> > entriesLocal;
    auto ptr = std::dynamic_pointer_cast<SparsePenaltyFunction>(it);
    assert(ptr);
    ptr->JacobianEntries(beta, entriesLocal);
    for( auto& jt : entriesLocal ) { entries.emplace_back(start+jt.row(), jt.col(), jt.value()); }
    start += it->outdim;
  }
  
  jac.resize(numTerms, InputDimension()); 
  jac.setFromTriplets(entries.begin(), entries.end());
}

