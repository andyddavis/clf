#ifndef COSTFUNCTION_HPP_
#define COSTFUNCTION_HPP_

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <MUQ/Optimization/CostFunction.h>

namespace clf {

/// A cost function that can be minimized using the Levenberg Marquardt algorithm
/**
The cost function is the sum of \f$m\f$ squared penalty functions \f$f_i: \mathbb{R}^{n} \mapsto \mathbb{R}^{d_i}\f$,
\f{equation*}{
C = \min_{\boldsymbol{\beta} \in \mathbb{R}^{n}} \sum_{i=1}^{m} \| f_i(\boldsymbol{\beta}) \|^{2}.
\f}
*/
template<typename MatrixType>
class CostFunction : public muq::Optimization::CostFunction {
public:

  /// Create a cost function with \f$m\f$ penalty functions that all have the same output dimension \f$d\f$
  /**
  @param[in] inputDimension The dimension of the input parameter \f$n\f$
  @param[in] numPenaltyFunctions The number of penalty functions \f$m\f$
  @param[in] outputDimension The output dimension \f$d\f$
  */
  inline CostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions, std::size_t const outputDimension) :
  muq::Optimization::CostFunction(Eigen::VectorXi::Constant(1, inputDimension)),
  inputDimension(inputDimension),
  numPenaltyFunctions(numPenaltyFunctions),
  numPenaltyTerms(numPenaltyFunctions*outputDimension),
  outputDimension(std::vector<std::pair<std::size_t, std::size_t> >({std::pair<std::size_t, std::size_t>(numPenaltyFunctions, outputDimension)}))
  {}

  /// Create a cost function with \f$m\f$ penalty terms that all have different output dimension
  /**
  @param[in] inputDimension The dimension of the input parameter \f$n\f$
  @param[in] numPenaltyFunctions The number of penalty functions \f$m\f$
  @param[in] outputDimensions Each component indicates there is <tt>outputDimension[i].first</tt> penalty functions with dimension <tt>outputDimension[i].second</tt>
  */
  inline CostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions, std::vector<std::pair<std::size_t, std::size_t> > const& outputDimension) :
  muq::Optimization::CostFunction(Eigen::VectorXi::Constant(1, inputDimension)),
  inputDimension(inputDimension),
  numPenaltyFunctions(numPenaltyFunctions),
  numPenaltyTerms(numPenaltyFunctions*NumPenaltyTerms(outputDimension)),
  outputDimension(outputDimension)
  {}

  virtual ~CostFunction() = default;

  /// The output dimension of the \f$i^{th}\f$ penalty function
  /**
  @param[in] ind The index of the penalty function
  \return The output dimension of the corresponding penalty function
  */
  inline std::size_t PenaltyFunctionOutputDimension(std::size_t const ind) const {
    std::size_t count = 0;
    for( const auto& it : outputDimension ) {
      count += it.first;
      if( ind<count ) { return it.second; }
    }
    assert(false);
    return std::numeric_limits<std::size_t>::max();
  }

  /// Evaluate the \f$i^{th}\f$ penalty function \f$f_i: \mathbb{R}^{n} \mapsto \mathbb{R}^{d_i}\f$
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The evaluation of the \f$i^{th}\f$ penalty function
  */
  inline Eigen::VectorXd PenaltyFunction(std::size_t const ind, Eigen::VectorXd const& beta) const {
    assert(beta.size()==inputDimension);
    assert(ind<numPenaltyFunctions);
    return PenaltyFunctionImpl(ind, beta);
  }

  /// Evaluate the Jacobian \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$
  */
  inline Eigen::MatrixXd PenaltyFunctionJacobian(std::size_t const ind, Eigen::VectorXd const& beta) const {
    assert(beta.size()==inputDimension);
    assert(ind<numPenaltyFunctions);
    const Eigen::MatrixXd jac = PenaltyFunctionJacobianImpl(ind, beta);
    assert(jac.cols()==inputDimension);
    return jac;
  }

  /// Evaluate the Jacobain \f$\nabla_{\beta} f_i(\beta)\f$ (which is an \f$n \times d_i\f$ matrix) using finite difference
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  @param[in] dbeta The \f$\Delta \beta\f$ used to compute finite difference approximations (defaults to \f$1e-8\f$)
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$
  */
  inline Eigen::MatrixXd PenaltyFunctionJacobianByFD(std::size_t const ind, Eigen::VectorXd const& beta, double const dbeta = 1.0e-8) const {
    assert(beta.size()==inputDimension);
    assert(ind<numPenaltyFunctions);
    const std::size_t outputDimension = PenaltyFunctionOutputDimension(ind);
    Eigen::MatrixXd jac(outputDimension, inputDimension);
    const Eigen::VectorXd cost = PenaltyFunction(ind, beta);

    Eigen::VectorXd betaPlus = beta;
    for( std::size_t i=0; i<inputDimension; ++i ) {
      betaPlus(i) += dbeta;
      const Eigen::VectorXd costp = PenaltyFunction(ind, betaPlus);
      betaPlus(i) -= dbeta;
      for( std::size_t d=0; d<outputDimension; ++d ) { jac(d, i) = (costp(d)-cost(d))/dbeta; }
    }

    return jac;
  }

  /// Evaluate each penalty function \f$f_i(\boldsymbol{\beta})\f$
  /**
  @param[in] beta The current parameter value
  \return The \f$i^{th}\f$ entry is the \f$i^{th}\f$ penalty function \f$f_i(\boldsymbol{\beta})\f$
  */
  inline Eigen::VectorXd CostVector(Eigen::VectorXd const& beta) const {
    assert(beta.size()==inputDimension);
    Eigen::VectorXd cost(numPenaltyTerms);
    std::size_t count = 0;
    std::size_t cnt = 0;
    for( std::size_t i=0; i<numPenaltyFunctions; ++i ) {
      if( i>=outputDimension[count].first ) { ++count; }
      cost.segment(cnt, outputDimension[count].second) = PenaltyFunction(i, beta);
      cnt += outputDimension[count].second;
    }
    return cost;
  }

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the penalty function \f$f_i\f$ with respect to the input parameters \f$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::PenaltyFunctionJacobianImpl to compute the Jacobian matrix.
  @param[in] beta The current parameter value
  @param[out] jac The Jacobian matrix
  */
  virtual void Jacobian(Eigen::VectorXd const& beta, MatrixType& jac) const = 0;

  /// Is this a quadratic cost function?
  /**
  Defaults to <tt>false</tt>, but can be overriden by children.
  \return <tt>true</tt>: The cost function is quadratic, <tt>false</tt>: The cost function is not quadratic
  */
  inline virtual bool IsQuadratic() const { return false; }

  /// The dimension of the input parameter \f$n\f$
  const std::size_t inputDimension;

  /// The number of penalty functions \f$m\f$
  const std::size_t numPenaltyFunctions;

  /// The total number of squared scalar terms in the cost function
  const std::size_t numPenaltyTerms;

protected:

  /// Evaluate the \f$i^{th}\f$ penalty function \f$f_i: \mathbb{R}^{n} \mapsto \mathbb{R}^{d_i}\f$
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The evaluation of the \f$i^{th}\f$ penalty function
  */
  virtual Eigen::VectorXd PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const = 0;

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta)\f$
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$
  */
  inline virtual Eigen::MatrixXd PenaltyFunctionJacobianImpl(std::size_t const ind, Eigen::VectorXd const& beta) const { return PenaltyFunctionJacobianByFD(ind, beta); }

  /// Compute the total cost by summing the squared penalty terms
  /**
  @param[in] input There is only one input: the input parameters \f$\beta\f$
  \return The total cost
  */
  inline virtual double CostImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& input) override {
    double cost = 0.0;
    for( std::size_t i=0; i<numPenaltyFunctions; ++i ) {
      const Eigen::VectorXd fi = PenaltyFunction(i, input[0]) ;
      cost += fi.dot(fi);
    }
    return cost;
  }

  /// Compute the gradient of the total cost by summing the gradient of the squared penalty terms
  /**
  @param[in] input There is only one input: the input parameters \f$\beta\f$
  \return The gradient of the total cost
  */
  inline virtual void GradientImpl(unsigned int const inputDimWrt, muq::Modeling::ref_vector<Eigen::VectorXd> const& input, Eigen::VectorXd const& sensitivity) override {
    assert(false);
    gradient = Eigen::VectorXd::Zero(inputDimension);
    for( std::size_t i=0; i<numPenaltyFunctions; ++i ) {
      gradient += 2.0*PenaltyFunction(i, input[0])*PenaltyFunctionJacobian(i, input[0]);
    }
  }

  /// Each component indicates there is <tt>outputDimension[i].first</tt> penalty functions with dimension <tt>outputDimension[i].second</tt>
  const std::vector<std::pair<std::size_t, std::size_t> > outputDimension;

private:

  /// Compute the total number of squared penalty terms in the cost function
  /**
  @param[in] outputDimensions Each component indicates there is <tt>outputDimension[i].first</tt> penalty functions with dimension <tt>outputDimension[i].second</tt>
  */
  inline static std::size_t NumPenaltyTerms(std::vector<std::pair<std::size_t, std::size_t> > const& outputDimension) {
    std::size_t numTerms = 0;
    for( const auto& it : outputDimension ) { numTerms += it.first*it.second; }
    return numTerms;
  }

};

} // namespace clf

#endif
