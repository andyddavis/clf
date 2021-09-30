#include "clf/Optimizer.hpp"

using namespace clf;

template<>
Eigen::VectorXd Optimizer<Eigen::MatrixXd>::SolveLinearSystem(Eigen::MatrixXd const& mat, Eigen::VectorXd const& rhs) const {
  return Eigen::VectorXd();
}

template<>
Eigen::VectorXd Optimizer<Eigen::SparseMatrix<double> >::SolveLinearSystem(Eigen::SparseMatrix<double> const& mat, Eigen::VectorXd const& rhs) const {
  return Eigen::VectorXd();
}
