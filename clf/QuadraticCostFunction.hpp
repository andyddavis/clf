#ifndef QUADRATICCOSTFUNCTION_HPP_
#define QUADRATICCOSTFUNCTION_HPP_

#include "clf/CostFunction.hpp"

namespace clf {

/// A clf::CostFunction with linear penalty functions
/**
The cost function is the some of squared penalty function \f$f_i\f$---
\f{equation*}{
C = \min_{\boldsymbol{\beta} \in \mathbb{R}^{n}} \sum_{i=1}^{m} f_i(\boldsymbol{\beta})^{2}.
\f}
In the case of a quadratic cost function the penalty functions \f$f_i\f$ are all linear with respect to \f$\beta\f$.
*/
template<typename MatrixType>
class QuadraticCostFunction : public CostFunction<MatrixType> {
public:

  /**
  @param[in] inputDimension The dimension of the input parameter \f$n\f$
  @param[in] numPenaltyFunctions The number of penalty functions \f$m\f$
  */
  inline QuadraticCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions) : CostFunction<MatrixType>(inputDimension, numPenaltyFunctions) {}

  virtual ~QuadraticCostFunction() = default;

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  The gradient should be independent of \f$\beta\f$ since \f$f_i\f$ is linear in this case. 
  @param[in] ind The index of the penalty function
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$
  */
  inline Eigen::VectorXd PenaltyFunctionGradient(std::size_t const ind) const {
    assert(ind<this->numPenaltyFunctions);
    const Eigen::VectorXd grad = PenaltyFunctionGradientImpl(ind);
    assert(grad.size()==this->inputDimension);
    return grad;
  }

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the penalty function \f$f_i\f$ with respect to the input parameters \f$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$. Since each penalty function is linear with respect to \f$\beta\f$, this matrix is independent of the parameter \f$\beta\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::PenaltyFunctionGradientImpl to compute the Jacobian matrix.
  @param[out] jac The Jacobian matrix
  */
  virtual void Jacobian(MatrixType& jac) const = 0;

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the sub-cost function \f$f_i\f$ with respect to the input parameters \$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$. Since each penalty function is linear with respect to \f$\beta\f$, this matrix is independent of the parameter \f$\beta\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::JacobianImpl to compute the Jacobian matrix.
  @param[in] beta The current parameter value
  @param[out] jac The Jacobian matrix
  */
  inline virtual void Jacobian(Eigen::VectorXd const& beta, MatrixType& jac) const final override { Jacobian(jac); }

  /// Is this a quadratic cost function? Yes, it is.
  /**
  \return <tt>true</tt>: The cost function is quadratic
  */
  virtual inline bool IsQuadratic() const final override { return true; }

protected:

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  The gradient should be independent of \f$\beta\f$ since \f$f_i\f$ is linear in this case. Therefore, override this function to just call the implementation without \f$\beta\f$.
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$
  */
  inline virtual Eigen::VectorXd PenaltyFunctionGradientImpl(std::size_t const ind, Eigen::VectorXd const& beta) const final override { return PenaltyFunctionGradientImpl(ind); }

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  This function computes the gradient using finite difference around \f$\beta=0\f$. More efficient gradient calculation can be implemented by children.
  @param[in] ind The index of the penalty function
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$
  */
  inline virtual Eigen::VectorXd PenaltyFunctionGradientImpl(std::size_t const ind) const { return this->PenaltyFunctionGradientByFD(ind, Eigen::VectorXd::Zero(this->inputDimension)); }

private:
};

/// A dense clf::QuadraticCostFunction
/**
The cost function is the some of squared penalty function \f$f_i\f$---
\f{equation*}{
C = \min_{\boldsymbol{\beta} \in \mathbb{R}^{n}} \sum_{i=1}^{m} f_i(\boldsymbol{\beta})^{2}.
\f}
In the case of a quadratic cost function the penalty functions \f$f_i\f$ are all linear with respect to \f$\beta\f$.
*/
class DenseQuadraticCostFunction : public QuadraticCostFunction<Eigen::MatrixXd> {
public:

  /**
  @param[in] inputDimension The dimension of the input parameter \f$n\f$
  @param[in] numPenaltyFunctions The number of penalty functions \f$m\f$
  */
  DenseQuadraticCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions);

  virtual ~DenseQuadraticCostFunction() = default;

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the penalty function \f$f_i\f$ with respect to the input parameters \f$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$. Since each penalty function is linear with respect to \f$\beta\f$, this matrix is independent of the parameter \f$\beta\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::PenaltyFunctionGradientImpl to compute the Jacobian matrix.
  @param[out] jac The Jacobian matrix
  */
  virtual void Jacobian(Eigen::MatrixXd& jac) const final override;

protected:

private:
};

/// A sparse clf::QuadraticCostFunction 
/**
The cost function is the some of squared penalty function \f$f_i\f$---
\f{equation*}{
C = \min_{\boldsymbol{\beta} \in \mathbb{R}^{n}} \sum_{i=1}^{m} f_i(\boldsymbol{\beta})^{2}.
\f}
In the case of a quadratic cost function the penalty functions \f$f_i\f$ are all linear with respect to \f$\beta\f$.
*/
class SparseQuadraticCostFunction : public QuadraticCostFunction<Eigen::SparseMatrix<double> > {
public:

  /**
  @param[in] inputDimension The dimension of the input parameter \f$n\f$
  @param[in] numPenaltyFunctions The number of penalty functions \f$m\f$
  */
  SparseQuadraticCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions);

  virtual ~SparseQuadraticCostFunction() = default;

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  @param[in] ind The index of the penalty function
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$, each entry holds the index and value of a non-zero entry
  */
  std::vector<std::pair<std::size_t, double> > PenaltyFunctionGradientSparse(std::size_t const ind) const;

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the sub-cost function \f$f_i\f$ with respect to the input parameters \$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$. Since each penalty function is linear with respect to \f$\beta\f$, this matrix is independent of the parameter \f$\beta\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::JacobianImpl to compute the Jacobian matrix.
  @param[out] jac The Jacobian matrix
  */
  virtual void Jacobian(Eigen::SparseMatrix<double>& jac) const final override;

protected:

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  The user can no longer override this function. They must override the sparse version.
  @param[in] ind The index of the penalty function
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$
  */
  virtual Eigen::VectorXd PenaltyFunctionGradientImpl(std::size_t const ind) const final override;

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  Default to using finite difference with \f$\beta=0\f$. By children can implement more efficient gradient computations
  @param[in] ind The index of the penalty function
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$, each entry holds the index and value of a non-zero entry
  */
  virtual std::vector<std::pair<std::size_t, double> > PenaltyFunctionGradientSparseImpl(std::size_t const ind) const;

  /// The sparsity tolerance ignores entries in the Jacobian that are less then this value 
  /**
  Defaults to \f$1.0e-14\f$
  */
  const double sparsityTol = 1.0e-14;

private:
};

} // namespace clf

#endif 
