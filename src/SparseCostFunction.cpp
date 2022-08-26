#include "clf/SparseCostFunction.hpp"

using namespace clf;

SparseCostFunction::SparseCostFunction(SparsePenaltyFunctions const& penaltyFunctions) : CostFunction<Eigen::SparseMatrix<double> >(penaltyFunctions) {}

SparseCostFunction::SparseCostFunction(std::shared_ptr<SparsePenaltyFunction> const& penaltyFunction) : CostFunction<Eigen::SparseMatrix<double> >(penaltyFunction) {}

Eigen::SparseMatrix<double> SparseCostFunction::Jacobian(Eigen::VectorXd const& beta) const { 
  assert(beta.size()==InputDimension());
  
  std::vector<Eigen::Triplet<double> > entries;
  std::size_t start = 0;
  for( const auto& it : penaltyFunctions ) {
    auto ptr = std::dynamic_pointer_cast<SparsePenaltyFunction>(it);
    assert(ptr);
    std::vector<Eigen::Triplet<double> > entriesLocal = ptr->JacobianEntries(beta);
    for( auto& jt : entriesLocal ) { entries.emplace_back(start+jt.row(), jt.col(), jt.value()); }
    start += it->OutputDimension();
  }
  
  Eigen::SparseMatrix<double> jac(numTerms, InputDimension()); 
  jac.setFromTriplets(entries.begin(), entries.end());
  return jac;
}

