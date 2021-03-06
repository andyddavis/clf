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
C = \min_{\beta \in \mathbb{R}^{n}} \sum_{i=1}^{m} \| f_i(\beta) \|^{2}.
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
  muq::Optimization::CostFunction(inputDimension),
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
  muq::Optimization::CostFunction(inputDimension),
  inputDimension(inputDimension),
  numPenaltyFunctions(numPenaltyFunctions),
  numPenaltyTerms(NumPenaltyTerms(outputDimension)),
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

  /// Evaluate the Jacobian \f$\nabla_{\beta} f_i(\beta) \in \mathbb{R}^{d_i \times n}\f$
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta) \in \mathbb{R}^{d_i \times n}\f$
  */
  inline Eigen::MatrixXd PenaltyFunctionJacobian(std::size_t const ind, Eigen::VectorXd const& beta) const {
    assert(beta.size()==inputDimension);
    assert(ind<numPenaltyFunctions);
    const Eigen::MatrixXd jac = PenaltyFunctionJacobianImpl(ind, beta);
    assert(jac.cols()==inputDimension);
    return jac;
  }

  /// The order for the finite difference approximation
  enum FDOrder {
    /// First order upward approximation 
    FIRST_UPWARD,

    /// First order backward approximation
    FIRST_DOWNWARD,

    /// Second order centered approximation
    SECOND,

    /// Fourth order centered approximation
    FOURTH,

    /// Sixth order centered approximation
    SIXTH
  };

  /// Evaluate the Jacobain \f$\nabla_{\beta} f_i(\beta) \in \mathbb{R}^{d_i \times n}\f$ using finite difference
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  @param[in] order The order of the finite difference approximation
  @param[in] dbeta The \f$\Delta \beta\f$ used to compute finite difference approximations (defaults to \f$1e-8\f$)
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta) \in \mathbb{R}^{d_i \times n}\f$
  */
  inline Eigen::MatrixXd PenaltyFunctionJacobianByFD(std::size_t const ind, Eigen::VectorXd const& beta, FDOrder const order = FIRST_UPWARD, double const dbeta = 1.0e-8) const {
    assert(beta.size()==inputDimension);
    assert(ind<numPenaltyFunctions);
    const std::size_t outputDimension = PenaltyFunctionOutputDimension(ind);
    Eigen::MatrixXd jac(outputDimension, inputDimension);
    
    // precompute if we need this
    Eigen::VectorXd cost;
    if( order==FDOrder::FIRST_UPWARD | order==FDOrder::FIRST_DOWNWARD ) { cost = PenaltyFunction(ind, beta); }

    Eigen::VectorXd betaFD = beta;
    for( std::size_t i=0; i<inputDimension; ++i ) {
      switch( order ) {
      case FDOrder::FIRST_UPWARD: {
	betaFD(i) += dbeta;
	const Eigen::VectorXd costp = PenaltyFunction(ind, betaFD);
	betaFD(i) -= dbeta;
	jac.col(i) = (costp-cost)/dbeta;
	break;
      }
      case FDOrder::FIRST_DOWNWARD: {
	betaFD(i) -= dbeta;
	const Eigen::VectorXd costm = PenaltyFunction(ind, betaFD);
	betaFD(i) += dbeta;
	jac.col(i) = (cost-costm)/dbeta;
	break;
      }
      case FDOrder::SECOND: {
	betaFD(i) -= dbeta;
	const Eigen::VectorXd costm = PenaltyFunction(ind, betaFD);
	betaFD(i) += 2.0*dbeta;
	const Eigen::VectorXd costp = PenaltyFunction(ind, betaFD);
	betaFD(i) -= dbeta;
	jac.col(i) = (costp-costm)/(2.0*dbeta);
	break;
      }
      case FDOrder::FOURTH: {
	betaFD(i) -= dbeta;
	const Eigen::VectorXd costm1 = PenaltyFunction(ind, betaFD);
	betaFD(i) -= dbeta;
	const Eigen::VectorXd costm2 = PenaltyFunction(ind, betaFD);
	betaFD(i) += 3.0*dbeta;
	const Eigen::VectorXd costp1 = PenaltyFunction(ind, betaFD);
	betaFD(i) += dbeta;
	const Eigen::VectorXd costp2 = PenaltyFunction(ind, betaFD);
	betaFD(i) -= 2.0*dbeta;
	jac.col(i) = ((costm2-costp2)/12.0+(2.0/3.0)*(costp1-costm1))/dbeta;
	break;
      }
      case FDOrder::SIXTH: {
	betaFD(i) -= dbeta;
	const Eigen::VectorXd costm1 = PenaltyFunction(ind, betaFD);
	betaFD(i) -= dbeta;
	const Eigen::VectorXd costm2 = PenaltyFunction(ind, betaFD);
	betaFD(i) -= dbeta;
	const Eigen::VectorXd costm3 = PenaltyFunction(ind, betaFD);
	betaFD(i) += 4.0*dbeta;
	const Eigen::VectorXd costp1 = PenaltyFunction(ind, betaFD);
	betaFD(i) += dbeta;
	const Eigen::VectorXd costp2 = PenaltyFunction(ind, betaFD);
	betaFD(i) += dbeta;
	const Eigen::VectorXd costp3 = PenaltyFunction(ind, betaFD);
	betaFD(i) -= 3.0*dbeta;
	jac.col(i) = ((costp3-costm3)/60.0+(3.0/20.0)*(costm2-costp2)+(3.0/4.0)*(costp1-costm1))/dbeta;
	break;
      }
      }
    }

    return jac;
  }

  /// Evaluate the Hessian \f$\nabla_{\beta}^2 f_i^{(j)}(\beta) \in \mathbb{R}^{n \times n}\f$ of the penalty function
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter \f$\beta \in \mathbb{R}^{n}\f$
  \return Each component is the Hessian of the \f$j^{th}\f$ couput of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta}^2 f_i^{(j)}(\beta) \in \mathbb{R}^{n \times n}\f$
  */
  inline std::vector<MatrixType> PenaltyFunctionHessian(std::size_t const ind, Eigen::VectorXd const& beta) const {
    assert(beta.size()==inputDimension);
    assert(ind<numPenaltyFunctions);

    const std::vector<MatrixType> hess = PenaltyFunctionHessianImpl(ind, beta);
    assert(hess.size()==PenaltyFunctionOutputDimension(ind));
    return hess;
  }

  /// Evaluate the Hessian \f$\nabla_{\beta}^2 f_i^{(j)}(\beta) \in \mathbb{R}^{n \times n}\f$ of the penalty function
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter \f$\beta \in \mathbb{R}^{n}\f$
  @param[in] order The order of the finite difference approximation
  @param[in] dbeta The \f$\Delta \beta\f$ used to compute finite difference approximations (defaults to \f$1e-8\f$)
  \return Each component is the Hessian of the \f$j^{th}\f$ couput of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta}^2 f_i^{(j)}(\beta) \in \mathbb{R}^{n \times n}\f$
  */
  virtual std::vector<MatrixType> PenaltyFunctionHessianByFD(std::size_t const ind, Eigen::VectorXd const& beta, FDOrder const order = FIRST_UPWARD, double const dbeta = 1.0e-8) const = 0;

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
  The Jacobian matrix is \f$J\f$ such that each row is the gradient of a penalty term. The Jacobian of the \f$i^{th}\f$ cost function makes up \f$d_i\f$ rows of the matrix \f$J\f$ ordered according to the ordering of \f$f_i\f$.

  This function resets the Jacobian to zero and then calls clf::CostFunction::PenaltyFunctionJacobian to compute the Jacobian matrix.
  \f{equation*}{
     J(\beta) = \left[ \begin{array}{ccc}
        --- & \nabla_{\beta} f_1(\beta) & --- \\
	--- & \vdots & --- \\
        --- & \nabla_{\beta} f_m(\beta) & --- \\
     \end{array} \right] \in \mathbb{R}^{\sum_{i=1}^{n} d_i \times n}
\f}
  @param[in] beta The current parameter value
  @param[out] jac The Jacobian matrix
  */
  virtual void Jacobian(Eigen::VectorXd const& beta, MatrixType& jac) const = 0;

  /// Compute the Hessian of the cost function
  /**
  The Hessian of the cost function is
  \f{equation*}{
  H = 2 \sum_{i=1}^{m} \left( \sum_{j=1}^{d_i} f_i^{(j)}(\beta) \nabla_{\beta}^2 f_i^{(j)}(\beta) + (\nabla_{\beta} f_i(\beta))^{\top} \nabla_{\beta} f_i(\beta) \right), 
  \f}
  where \f$\nabla_{\beta}^2 f_i^{(j)}(\beta)\f$ is the Hessian of the \f$j^{th}\f$ output of \f$f_i\f$. Alternatively, we could compute the Gauss-Newton approximation
  \f{equation*}{
  H = 2 \sum_{i=1}^{m} (\nabla_{\beta} f_i(\beta))^{\top} \nabla_{\beta} f_i(\beta).
  \f}
  @param[in] beta The current parameter value
  @param[out] hess The Hessian matrix
  @param[in] gn <tt>true</tt>: Compute the Gauss-Newton Hessian, <tt>false</tt> (default): Compute the full Hessian 
  */
  inline virtual void Hessian(Eigen::VectorXd const& beta, MatrixType& hess, bool const gn = false) { 
    // compute the Gauss-Newton Hessian 
    GaussNewtonHessian(beta, hess);
    if( gn ) { return; }
    HessianGivenGaussNewtonHessian(beta, hess);
  }

  /// Compute the Hessian of the cost function given that we have already computed the Jacobian
  /**
  The Hessian of the cost function is
  \f{equation*}{
  H = 2 \sum_{i=1}^{m} \left( \sum_{j=1}^{d_i} f_i^{(j)}(\beta) \nabla_{\beta}^2 f_i^{(j)}(\beta) + (\nabla_{\beta} f_i(\beta))^{\top} \nabla_{\beta} f_i(\beta) \right), 
  \f}
  where \f$\nabla_{\beta}^2 f_i^{(j)}(\beta)\f$ is the Hessian of the \f$j^{th}\f$ output of \f$f_i\f$. Alternatively, we could compute the Gauss-Newton approximation
  \f{equation*}{
  H = 2 \sum_{i=1}^{m} (\nabla_{\beta} f_i(\beta))^{\top} \nabla_{\beta} f_i(\beta).
  \f}
  @param[in] beta The current parameter value
  @param[in] jac The pre-computed Jacobian
  @param[out] hess The Hessian matrix
  @param[in] gn <tt>true</tt>: Compute the Gauss-Newton Hessian, <tt>false</tt> (default): Compute the full Hessian 
  */
  inline virtual void Hessian(Eigen::VectorXd const& beta, MatrixType const& jac, MatrixType& hess, bool const gn = false) { 
    // compute the Gauss-Newton Hessian 
    GaussNewtonHessianGivenJacobian(jac, hess);
    if( gn ) { return; }
    HessianGivenGaussNewtonHessian(beta, hess);
  }

  /// Compute the Gauss-Newton Hessian of the cost function
  /**
  We compute the Gauss-Newton approximation to the Hessian
  \f{equation*}{
  H = 2 \sum_{i=1}^{m} (\nabla_{\beta} f_i(\beta))^{\top} \nabla_{\beta} f_i(\beta).
  \f}
  @param[in] beta The current parameter value
  @param[out] gnHess The Gauss-Newton Hessian matrix
  */
  inline void GaussNewtonHessian(Eigen::VectorXd const& beta, MatrixType& gnHess) const {
    Jacobian(beta, gnHess);
    gnHess = 2.0*gnHess.transpose()*gnHess;
  }

  /// Compute the Gauss-Newton Hessian of the cost function given that we have already computed the Jacobian
  /**
  We compute the Gauss-Newton approximation to the Hessian
  \f{equation*}{
  H = 2 \sum_{i=1}^{m} (\nabla_{\beta} f_i(\beta))^{\top} \nabla_{\beta} f_i(\beta).
  \f}
  @param[in] jac The pre-computed Jacobian
  @param[out] gnHess The Gauss-Newton Hessian matrix
  */
  inline void GaussNewtonHessianGivenJacobian(MatrixType const& jac, MatrixType& gnHess) const {
    gnHess = 2.0*jac.transpose()*jac;
  }

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

  /// Evaluate the gradient \f$\nabla_{\beta} f_i(\beta) \in \mathbb{R}^{d_i \times n}\f$
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta) \in \mathbb{R}^{d_i \times n}\f$
  */
  inline virtual Eigen::MatrixXd PenaltyFunctionJacobianImpl(std::size_t const ind, Eigen::VectorXd const& beta) const { return PenaltyFunctionJacobianByFD(ind, beta); }

  /// Evaluate the Hessian \f$\nabla_{\beta}^2 f_i^{(j)}(\beta) \in \mathbb{R}^{n \times n}\f$ of the penalty function
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter \f$\beta \in \mathbb{R}^{n}\f$
  \return Each component is the Hessian of the \f$j^{th}\f$ couput of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta}^2 f_i^{(j)}(\beta) \in \mathbb{R}^{n \times n}\f$
  */
  inline virtual std::vector<MatrixType> PenaltyFunctionHessianImpl(std::size_t const ind, Eigen::VectorXd const& beta) const { return PenaltyFunctionHessianByFD(ind, beta); }

  /// Compute the total cost by summing the squared penalty terms
  /**
  @param[in] input There is only one input: the input parameters \f$\beta\f$
  \return The total cost
  */
  //inline virtual double CostImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& input) override {
  inline virtual double Cost() override {
    double cost = 0.0;
    for( std::size_t i=0; i<numPenaltyFunctions; ++i ) {
      const Eigen::VectorXd fi = PenaltyFunction(i, x) ;
      cost += fi.dot(fi);
    }
    return cost;
  }

  /// Compute the gradient of the total cost by summing the gradient of the squared penalty terms
  /**
  @param[in] input There is only one input: the input parameters \f$\beta\f$
  \return The gradient of the total cost
  */
  //inline virtual void GradientImpl(unsigned int const inputDimWrt, muq::Modeling::ref_vector<Eigen::VectorXd> const& input, Eigen::VectorXd const& sensitivity) override {
  inline virtual Eigen::VectorXd Gradient() override {
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(inputDimension);
    for( std::size_t i=0; i<numPenaltyFunctions; ++i ) { gradient += 2.0*PenaltyFunctionJacobian(i, x).transpose()*PenaltyFunction(i, x); }
    return gradient;
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

  /// Compute the Hessian of the cost function
  /**
  \return The Hessian of the cost function (not the Guass-Newton Hessian).
  */
  inline virtual Eigen::MatrixXd Hessian() override { 
    MatrixType hess;
    Hessian(x, hess, false); 
    return (Eigen::MatrixXd)hess;
  }

  /// Compute the Hessian of the cost function given that we have already computed the Gauss-Newton Hessian
  /**
  The Hessian of the cost function is
  \f{equation*}{
  H = 2 \sum_{i=1}^{m} \left( \sum_{j=1}^{d_i} f_i^{(j)}(\beta) \nabla_{\beta}^2 f_i^{(j)}(\beta) + (\nabla_{\beta} f_i(\beta))^{\top} \nabla_{\beta} f_i(\beta) \right), 
  \f}
  where \f$\nabla_{\beta}^2 f_i^{(j)}(\beta)\f$ is the Hessian of the \f$j^{th}\f$ output of \f$f_i\f$. Alternatively, we could compute the Gauss-Newton approximation
  \f{equation*}{
  H = 2 \sum_{i=1}^{m} (\nabla_{\beta} f_i(\beta))^{\top} \nabla_{\beta} f_i(\beta).
  \f}
  @param[in] beta The current parameter value
  @param[in,out] hess In: The Gauss-NewtonHessian matrix, out: The Hessian matrix
  */
  inline virtual void HessianGivenGaussNewtonHessian(Eigen::VectorXd const& beta, MatrixType& hess) { 
    // compute the cost vector 
    const Eigen::VectorXd cost = 2.0*CostVector(beta);

    std::size_t count = 0;
    for( std::size_t i=0; i<numPenaltyFunctions; ++i ) { 
      const std::vector<MatrixType> hessi = PenaltyFunctionHessian(i, beta);
      for( std::size_t j=0; j<hessi.size(); ++j ) { hess += cost(count+j)*hessi[j]; }
      count += hessi.size();
    }
  }

};

} // namespace clf

#endif
