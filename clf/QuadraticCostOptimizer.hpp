#ifndef QUADRATICCOSTOPTIMIZER_HPP_
#define QUADRATICCOSTOPTIMIZER_HPP_

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

  /**
  @param[in] cost The cost function that we need to minimize (clf::CostFunction::IsQuadratic must be true)
  @param[in] pt Options for the algorithm
  */
  inline QuadraticCostOptimizer(std::shared_ptr<CostFunction<MatrixType> > const& cost, boost::property_tree::ptree const& pt) :
  Optimizer<MatrixType>(cost, pt)
  {}

  virtual ~QuadraticCostOptimizer() = default;

  /// Minimize the cost function
  /**
  @param[in,out] beta In: The initial guess for the optimization algorithm; Out: The paramter values that minimize the cost function
  \return First: Information about convergence or failure, Second: The current cost
  */
  inline virtual std::pair<Optimization::Convergence, double> Minimize(Eigen::VectorXd& beta) override {
    return  std::pair<Optimization::Convergence, double>(Optimization::Convergence::CONVERGED, 0.0);
  }

private:
};

typedef QuadraticCostOptimizer<Eigen::MatrixXd> DenseQuadraticCostOptimizer;
typedef QuadraticCostOptimizer<Eigen::SparseMatrix<double> > SparseQuadraticCostOptimizer;

} // namespace clf

#endif
