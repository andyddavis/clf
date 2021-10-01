#include "clf/Optimizer.hpp"

#include <Eigen/QR>
#include <Eigen/LU>

using namespace clf;

template<>
Eigen::VectorXd Optimizer<Eigen::MatrixXd>::SolveLinearSystem(Eigen::MatrixXd const& mat, Eigen::VectorXd const& rhs) const {
  assert(mat.rows()==rhs.size());
  if( linSolver==Optimizer::QR ) { 
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver(mat);
    assert(solver.info()==Eigen::Success);
    return SolveLinearSystemQR(solver, rhs);
  }

  Eigen::PartialPivLU<Eigen::MatrixXd> solver(mat);
  return solver.solve(rhs);
}

template<>
Eigen::VectorXd Optimizer<Eigen::SparseMatrix<double> >::SolveLinearSystem(Eigen::SparseMatrix<double> const& mat, Eigen::VectorXd const& rhs) const {
  assert(mat.rows()==rhs.size());
  if( linSolver==Optimizer::QR ) { 
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver(mat);
    assert(solver.info()==Eigen::Success);
    return SolveLinearSystemQR(solver, rhs);
  }

  Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver(mat);
  return solver.solve(rhs);
}
