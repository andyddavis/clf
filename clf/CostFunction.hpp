#ifndef COSTFUNCTION_HPP_
#define COSTFUNCTION_HPP_

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace clf {

/// A cost function that can be minimized using the Levenberg Marquardt algorithm
/**
The cost function has the for
\f{equation*}{
C = \min_{\boldsymbol{\beta} \in \mathbb{R}^{n}} \sum_{i=1}^{m} f_i(\boldsymbol{\beta})^{2}
\f}
*/
template<typename MatrixType>
class CostFunction {
public:

  /**
  @param[in] inDim The dimension of the input parameter \f$n\f$
  @param[in] valDim The number of sub-cost functions \f$m\f$
  */
  inline CostFunction(std::size_t const inDim, std::size_t const valDim) : inDim(inDim), valDim(valDim) {}

  virtual ~CostFunction() = default;

  /// Evaluate each sub-cost function \f$f_i(\boldsymbol{\beta})\f$
  /**
  @param[in] beta The current parameter value
  \return The \f$i^{th}\f$ entry is the \f$i^{th}\f$ sub-cost function \f$f_i(\boldsymbol{\beta})\f$
  */
  inline Eigen::VectorXd Cost(Eigen::VectorXd const& beta) const {
    assert(beta.size()==inDim);
    const Eigen::VectorXd cost = CostImpl(beta);
    assert(cost.size()==valDim);
    return cost;
  }

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the sub-cost function \f$f_i\f$ with respect to the input parameters \$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::JacobianImpl to compute the Jacobian matrix.
  @param[in] beta The current parameter value
  @param[out] jac The Jacobian matrix
  */
  virtual void Jacobian(Eigen::VectorXd const& beta, MatrixType& jac) const = 0;

  /// The dimension of the input parameter \f$n\f$
  const std::size_t inDim;

  /// The number of sub-cost functions \f$m\f$
  const std::size_t valDim;

protected:

  /// Evaluate each sub-cost function \f$f_i(\boldsymbol{\beta})\f$
  /**
  Must be implemented by a child to actually compute the cost.
  @param[in] beta The current parameter value
  \return The \f$i^{th}\f$ entry is the \f$i^{th}\f$ sub-cost function \f$f_i(\boldsymbol{\beta})\f$
  */
  virtual Eigen::VectorXd CostImpl(Eigen::VectorXd const& beta) const = 0;

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the sub-cost function \f$f_i\f$ with respect to the input parameters \$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$.

  Must be implemented by a child to actually compute the Jacobian.
  @param[in] beta The current parameter value
  @param[out] jac The Jacobian matrix
  */
  virtual void JacobianImpl(Eigen::VectorXd const& beta, MatrixType& jac) const = 0;

private:
};

class DenseCostFunction : public CostFunction<Eigen::MatrixXd> {
public:
  /**
  @param[in] inDim The dimension of the input parameter \f$n\f$
  @param[in] valDim The number of sub-cost functions \f$m\f$
  */
  inline DenseCostFunction(std::size_t const inDim, std::size_t const valDim) : CostFunction(inDim, valDim) {}

  virtual ~DenseCostFunction() = default;

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the sub-cost function \f$f_i\f$ with respect to the input parameters \$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::JacobianImpl to compute the Jacobian matrix.
  @param[in] beta The current parameter value
  @param[out] jac The Jacobian matrix
  */
  inline virtual void Jacobian(Eigen::VectorXd const& beta, Eigen::MatrixXd& jac) const final override {
    jac = Eigen::MatrixXd::Zero(valDim, inDim);
    JacobianImpl(beta, jac);
  }

private:
};

class SparseCostFunction : public CostFunction<Eigen::SparseMatrix<double> > {
public:
  /**
  @param[in] inDim The dimension of the input parameter \f$n\f$
  @param[in] valDim The number of sub-cost functions \f$m\f$
  */
  inline SparseCostFunction(std::size_t const inDim, std::size_t const valDim) : CostFunction(inDim, valDim) {}

  virtual ~SparseCostFunction() = default;

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the sub-cost function \f$f_i\f$ with respect to the input parameters \$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::JacobianImpl to compute the Jacobian matrix.
  @param[in] beta The current parameter value
  @param[out] jac The Jacobian matrix
  */
  inline virtual void Jacobian(Eigen::VectorXd const& beta, Eigen::SparseMatrix<double>& jac) const final override {
    // resize the jacobian---this sets every entry to zero, but does not free the memory
    jac.resize(valDim, inDim);
    JacobianImpl(beta, jac);
    jac.makeCompressed();
  }
private:
};
} // namespace clf

#endif
