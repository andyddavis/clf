#ifndef QUADRATICCOSTOPTIMIZER_HPP_
#define QUADRATICCOSTOPTIMIZER_HPP_

#include <Eigen/QR>
#include <Eigen/LU>

#include "clf/Optimizer.hpp"

namespace clf {

/// Minimize a cost function with the form \f$J(p) = \| A p - f \|^2\f$
/**
Solve the minimization problem
\f{equation*}{
\hat{p} = \mbox{arg min}_{p \in \mathbb{R}^{q}} J(p) = \| A p - f \|^2,
\f}
where \f$A \in \mathbb{R}^{m \times q}\f$ and \f$f \in \mathbb{R}^{m}\f$. This requires us to solve the linear system \f$A^{\top} A p = A^{\top} f\f$. If \f$A^{\top} A\f$ is not full rank, we solve this problem in the least-squares sense (using the pseudo-inverse).
*/
template<typename MatrixType>
class QuadraticCostOptimizer : public Optimizer<MatrixType> {
public:

  /// The type for the LU solver
  typedef typename std::conditional<std::is_same<Eigen::MatrixXd, MatrixType>::value, Eigen::PartialPivLU<Eigen::MatrixXd>, Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > >::type SolverLU;

  /// The type for the QR solver
  typedef typename std::conditional<std::is_same<Eigen::MatrixXd, MatrixType>::value, Eigen::ColPivHouseholderQR<Eigen::MatrixXd>, Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > >::type SolverQR;

  /**
  @param[in] cost The cost function that we need to minimize (clf::CostFunction::IsQuadratic must be true)
  @param[in] pt Options for the algorithm
  */
  inline QuadraticCostOptimizer(std::shared_ptr<CostFunction<MatrixType> > const& cost, boost::property_tree::ptree const& pt) :
  Optimizer<MatrixType>(cost, pt)
  {
    // make sure the cost function is quadratic 
    assert(cost->IsQuadratic());
    
    // construct the jacobian matrix (this should be independent of the parameter values, we can can just pass a dummy vector)
    MatrixType jac;
    cost->Jacobian(Eigen::VectorXd(cost->inputDimension), jac);
    resid = -jac.transpose()*this->cost->CostVector(Eigen::VectorXd::Zero(cost->inputDimension));

    // comptue the matrix decomposition based on which solver we are using
    if( this->linSolver==Optimization::LinearSolver::QR ) { solverQR.emplace(jac.transpose()*jac); } else { solverLU.emplace(jac.transpose()*jac); }
  }

  virtual ~QuadraticCostOptimizer() = default;

  /// Minimize the cost function
  /**
  @param[in,out] beta In: The initial guess for the optimization algorithm; Out: The paramter values that minimize the cost function
  \return First: Information about convergence or failure, Second: The current cost
  */
  inline virtual std::pair<Optimization::Convergence, double> Minimize(Eigen::VectorXd& beta) override {
    beta = ( this->linSolver==Optimization::LinearSolver::QR? this->SolveLinearSystemQR(*solverQR, resid) : solverLU->solve(resid) );
    return std::pair<Optimization::Convergence, double>(Optimization::Convergence::CONVERGED, 0.0);
  }

private:

  /// The Jacobian transpose times the residual computed using the guess \f$\beta = 0\f$
  /**
  Since the cost function is quadratic, we should be able to compute the exact solution using any guess. Therefore, we precompute the residual with \f$beta = 0\f$.
  */
  Eigen::VectorXd resid;
  
  /// The LU solver used to compute the optimal solution 
  /**
  The decomposition is precomputed at construction if we are using an LU solve.
  */
  std::optional<SolverLU> solverLU;

  /// The QR solver used to compute the optimal solution
  /**
  The decomposition is precomputed at construction if we are using an QR solve.
  */
  std::optional<SolverQR> solverQR;
};

typedef QuadraticCostOptimizer<Eigen::MatrixXd> DenseQuadraticCostOptimizer;
typedef QuadraticCostOptimizer<Eigen::SparseMatrix<double> > SparseQuadraticCostOptimizer;

} // namespace clf

#endif
