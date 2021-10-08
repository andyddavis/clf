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
\hat{p} = \mbox{arg min}_{p \in \mathbb{R}^{q}} \sum_{i=1}^{n} \|f_i(p)\|^2
\f}
where all \f$f_i\f$ are a linear functions of the parameter \f$p\f$. The residual given a parameter value \f$p_0\f$ is \f$r(p) = f(p_0) + J (p-p_0)\f$ such that \f$f(p_0)\f$ is computed by clf::CostFunction::CostVector, the matrix \f$J\f$ is computed by clf::CostFunction::Jacobian, and the cost is \f$r(p) \cdot r(p)\f$. Therefore, we solve \f$J^{\top} J \delta = - J^{\top} f(p_0)\f$ and set the optimal parameter value \f$\hat{p} = p_0 + \delta\f$.
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
    cost->Jacobian(Eigen::VectorXd(cost->inputDimension), jac);
    //resid = -jac.transpose()*this->cost->CostVector(Eigen::VectorXd::Zero(cost->inputDimension));

    // comptue the matrix decomposition based on which solver we are using
    if( this->linSolver==Optimization::LinearSolver::QR ) { solverQR.emplace(jac.transpose()*jac); } else { solverLU.emplace(jac.transpose()*jac); }
  }

  virtual ~QuadraticCostOptimizer() = default;

  /// Minimize the cost function
  /**
  Solve the minimization problem
  \f{equation*}{
  \hat{p} = \mbox{arg min}_{p \in \mathbb{R}^{q}} \sum_{i=1}^{n} \|f_i(p)\|^2
  \f}
  where all \f$f_i\f$ are a linear functions of the parameter \f$p\f$. The residual given a parameter value \f$p_0\f$ is \f$r(p) = f(p_0) + J (p-p_0)\f$ such that \f$f(p_0)\f$ is computed by clf::CostFunction::CostVector, the matrix \f$J\f$ is computed by clf::CostFunction::Jacobian, and the cost is \f$r(p) \cdot r(p)\f$. Therefore, we solve \f$J^{\top} J \delta = - J^{\top} f(p_0)\f$ and set the optimal parameter value \f$\hat{p} = p_0 + \delta\f$.
  @param[in,out] beta In: The initial guess for the optimization algorithm; Out: The paramter values that minimize the cost function
  \return First: Information about convergence or failure, Second: The current cost
  */
  inline virtual std::pair<Optimization::Convergence, double> Minimize(Eigen::VectorXd& beta) override {
    const Eigen::VectorXd resid = -jac.transpose()*this->cost->CostVector(beta);
    beta += ( this->linSolver==Optimization::LinearSolver::QR? this->SolveLinearSystemQR(*solverQR, resid) : solverLU->solve(resid) );
    return std::pair<Optimization::Convergence, double>(Optimization::Convergence::CONVERGED, 0.0);
  }

private:

  /// The Jacobian matrix \f$J\f$ stored so that we can compute \f$-J^{top} f(p_0)\f$
  MatrixType jac;

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
