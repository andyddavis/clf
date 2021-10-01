#ifndef COSTFUNCTION_HPP_
#define COSTFUNCTION_HPP_

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <MUQ/Optimization/CostFunction.h>

namespace clf {

/// A cost function that can be minimized using the Levenberg Marquardt algorithm
/**
The cost function is the some of squared penalty function \f$f_i\f$---
\f{equation*}{
C = \min_{\boldsymbol{\beta} \in \mathbb{R}^{n}} \sum_{i=1}^{m} f_i(\boldsymbol{\beta})^{2}
\f}
*/
template<typename MatrixType>
class CostFunction : public muq::Optimization::CostFunction {
public:

  /**
  @param[in] inputDimension The dimension of the input parameter \f$n\f$
  @param[in] numPenaltyFunctions The number of penalty functions \f$m\f$
  */
  inline CostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions) :
  muq::Optimization::CostFunction(Eigen::VectorXi::Constant(1, inputDimension)),
  inputDimension(inputDimension),
  numPenaltyFunctions(numPenaltyFunctions)
  {}

  virtual ~CostFunction() = default;

  /// Evaluate the \f$i^{th}\f$ penalty function \f$f_i(\beta)\f$
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The evaluation of the \f$i^{th}\f$ penalty function
  */
  inline double PenaltyFunction(std::size_t const ind, Eigen::VectorXd const& beta) const {
    assert(beta.size()==inputDimension);
    assert(ind<numPenaltyFunctions);
    return PenaltyFunctionImpl(ind, beta);
  }

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$
  */
  inline Eigen::VectorXd PenaltyFunctionGradient(std::size_t const ind, Eigen::VectorXd const& beta) const {
    assert(beta.size()==inputDimension);
    assert(ind<numPenaltyFunctions);
    const Eigen::VectorXd grad = PenaltyFunctionGradientImpl(ind, beta);
    assert(grad.size()==inputDimension);
    return grad;
  }

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$ using finite difference
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  @param[in] dbeta The \f$\Delta \beta\f$ used to compute finite difference approximations (defaults to \f$1e-8\f$)
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$
  */
  inline Eigen::VectorXd PenaltyFunctionGradientByFD(std::size_t const ind, Eigen::VectorXd const& beta, double const dbeta = 1.0e-8) const {
    assert(beta.size()==inputDimension);
    assert(ind<numPenaltyFunctions);
    Eigen::VectorXd grad(inputDimension);
    const double cost = PenaltyFunction(ind, beta);

    Eigen::VectorXd betaPlus = beta;
    for( std::size_t i=0; i<inputDimension; ++i ) {
      betaPlus(i) += dbeta;
      if( i>0 ) { betaPlus(i-1) -= dbeta; }
      const double costp = PenaltyFunction(ind, betaPlus);
      grad(i) = (costp-cost)/dbeta;
    }

    return grad;
  }

  /// Evaluate each penalty function \f$f_i(\boldsymbol{\beta})\f$
  /**
  @param[in] beta The current parameter value
  \return The \f$i^{th}\f$ entry is the \f$i^{th}\f$ penalty function \f$f_i(\boldsymbol{\beta})\f$
  */
  inline Eigen::VectorXd CostVector(Eigen::VectorXd const& beta) const {
    assert(beta.size()==inputDimension);
    Eigen::VectorXd cost(numPenaltyFunctions);
    for( std::size_t i=0; i<numPenaltyFunctions; ++i ) { cost(i) = PenaltyFunction(i, beta); }
    return cost;
  }

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the penalty function \f$f_i\f$ with respect to the input parameters \f$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::PenaltyFunctionGradientImpl to compute the Jacobian matrix.
  @param[in] beta The current parameter value
  @param[out] jac The Jacobian matrix
  */
  virtual void Jacobian(Eigen::VectorXd const& beta, MatrixType& jac) const = 0;

  /// Is this a quadratic cost function?
  /**
  Defaults to <tt>false</tt>, but can be overriden by children. 
  \return <tt>true</tt>: The cost function is quadratic, <tt>false</tt>: The cost function is not quadratic
  */
  virtual bool IsQuadratic() const = 0;

  /// The dimension of the input parameter \f$n\f$
  const std::size_t inputDimension;

  /// The number of penalty functions \f$m\f$
  const std::size_t numPenaltyFunctions;

protected:

  /// Evaluate the \f$i^{th}\f$ penalty function \f$f_i(\beta)\f$
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The evaluation of the \f$i^{th}\f$ penalty function
  */
  virtual double PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const = 0;

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$
  */
  inline virtual Eigen::VectorXd PenaltyFunctionGradientImpl(std::size_t const ind, Eigen::VectorXd const& beta) const { return PenaltyFunctionGradientByFD(ind, beta); }

  /// Compute the total cost by summing the squared penalty terms 
  /**
  @param[in] input There is only one input: the input parameters \f$\beta\f$
  \return The total cost 
  */
  inline virtual double CostImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& input) override {
    double cost = 0.0;
    for( std::size_t i=0; i<numPenaltyFunctions; ++i ) {
      const double fi = PenaltyFunction(i, input[0]);
      cost += fi*fi;
    }
    return cost;
  }

  /// Compute the gradient of the total cost by summing the gradient of the squared penalty terms 
  /**
  @param[in] input There is only one input: the input parameters \f$\beta\f$
  \return The gradient of the total cost 
  */
  inline virtual void GradientImpl(unsigned int const inputDimWrt, muq::Modeling::ref_vector<Eigen::VectorXd> const& input, Eigen::VectorXd const& sensitivity) override {
    gradient = Eigen::VectorXd::Zero(inputDimension);
    for( std::size_t i=0; i<numPenaltyFunctions; ++i ) {
      gradient += 2.0*PenaltyFunction(i, input[0])*PenaltyFunctionGradient(i, input[0]);
    }
  }

private:
};

/// A cost function using a dense Jacobian matrix 
class DenseCostFunction : public CostFunction<Eigen::MatrixXd> {
public:
  /**
  @param[in] inputDimension The dimension of the input parameter \f$n\f$
  @param[in] numPenaltyFunctions The number of sub-cost functions \f$m\f$
  */
  inline DenseCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions) : CostFunction(inputDimension, numPenaltyFunctions) {}

  virtual ~DenseCostFunction() = default;

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the sub-cost function \f$f_i\f$ with respect to the input parameters \$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::JacobianImpl to compute the Jacobian matrix.
  @param[in] beta The current parameter value
  @param[out] jac The Jacobian matrix
  */
  inline virtual void Jacobian(Eigen::VectorXd const& beta, Eigen::MatrixXd& jac) const final override {
    jac.resize(numPenaltyFunctions, inputDimension);
    for( std::size_t i=0; i<numPenaltyFunctions; ++i ) { jac.row(i) = PenaltyFunctionGradient(i, beta); }
  }

private:
};

/// A cost function using a sparse Jacobian matrix 
class SparseCostFunction : public CostFunction<Eigen::SparseMatrix<double> > {
public:
  /**
  @param[in] inputDimension The dimension of the input parameter \f$n\f$
  @param[in] numPenaltyFunctions The number of sub-cost functions \f$m\f$
  */
  inline SparseCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions) : CostFunction(inputDimension, numPenaltyFunctions) {}

  virtual ~SparseCostFunction() = default;

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$, each entry holds the index and value of a non-zero entry
  */
  inline std::vector<std::pair<std::size_t, double> > PenaltyFunctionGradientSparse(std::size_t const ind, Eigen::VectorXd const& beta) const {
    assert(beta.size()==inputDimension);
    assert(ind<numPenaltyFunctions);
    const std::vector<std::pair<std::size_t, double> > grad = PenaltyFunctionGradientSparseImpl(ind, beta);
    assert(grad.size()<=inputDimension);
    return grad;
  }

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the sub-cost function \f$f_i\f$ with respect to the input parameters \$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::JacobianImpl to compute the Jacobian matrix.
  @param[in] beta The current parameter value
  @param[out] jac The Jacobian matrix
  */
  inline virtual void Jacobian(Eigen::VectorXd const& beta, Eigen::SparseMatrix<double>& jac) const final override {
    // resize the jacobian---this sets every entry to zero, but does not free the memory
    jac.resize(numPenaltyFunctions, inputDimension);
    std::vector<Eigen::Triplet<double> > triplets;
    for( std::size_t i=0; i<numPenaltyFunctions; ++i ) {
      const std::vector<std::pair<std::size_t, double> > rowi = PenaltyFunctionGradientSparse(i, beta);
      for( const auto& it : rowi ) { triplets.emplace_back(i, it.first, it.second); }
    }
    jac.setFromTriplets(triplets.begin(), triplets.end());
    jac.makeCompressed();
  }

protected:

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  The user can no longer override this function. They must override the sparse version.
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$
  */
  inline virtual Eigen::VectorXd PenaltyFunctionGradientImpl(std::size_t const ind, Eigen::VectorXd const& beta) const final override {
    const std::vector<std::pair<std::size_t, double> > sparseGrad = PenaltyFunctionGradientSparseImpl(ind, beta);

    Eigen::VectorXd grad = Eigen::VectorXd::Zero(inputDimension);
    for( const auto& it : sparseGrad ) { grad(it.first) = it.second; }
    return grad;
  }

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  Default to using finite difference.
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$, each entry holds the index and value of a non-zero entry
  */
  inline virtual std::vector<std::pair<std::size_t, double> > PenaltyFunctionGradientSparseImpl(std::size_t const ind, Eigen::VectorXd const& beta) const {
    const Eigen::VectorXd grad = PenaltyFunctionGradientByFD(ind, beta);
    std::vector<std::pair<std::size_t, double> > sparseGrad;
    for( std::size_t i=0; i<inputDimension; ++i ) {
      if( std::abs(grad(i))>sparsityTol ) { sparseGrad.emplace_back(i, grad(i)); }
    }
    return sparseGrad;
  }

  /// The sparsity tolerance ignores entries in the Jacobian that are less then this value 
  /**
  Defaults to \f$1.0e-14\f$
  */
  const double sparsityTol = 1.0e-14;

private:
};
} // namespace clf

#endif
