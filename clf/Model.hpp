#ifndef MODEL_HPP_
#define MODEL_HPP_

#include <iostream>

#include <boost/property_tree/ptree.hpp>

#include <Eigen/Core>

#include "clf/ModelExceptions.hpp"
#include "clf/BasisFunctions.hpp"

namespace clf {

/// Implements a model that is given to a clf::SupportPoint
/**
Each support point is associated with a model
\f{equation*}{
\mathcal{L}(u) = f,
\f}
where \f$u:\Omega \mapsto \mathbb{R}^{m}\f$ (and \f$\Omega \subseteq \mathbb{R}^{d}\f$) is the function that we are trying to represent with local functions such that \f$u(x) \approx \Phi_{\hat{x}}(x) p\f$ near \f$\hat{x}\f$. The matrix \f$\Phi_{\hat{x}}(x)\f$ is the matrix of basis functions associated with the clf::SupportPoint at \f$\hat{x}\f$. 

This class allows the user to implement \f$\mathcal{L}\f$ and/or \f$f\f$. 

<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"InputDimension"   | <tt>std::size_t</tt> | <tt>1</tt> | The input dimension \f$d\f$. |
"OutputDimension"   | <tt>std::size_t</tt> | <tt>1</tt> | The output dimension \f$m\f$. |
"FiniteDifferenceStep"   | <tt>double</tt> | <tt>1.0e-6</tt> | The parameter used to approximate derivatives with finite difference. |
*/
class Model {
public:

  /// Construct a linear model given the in/output dimension 
  /**
  @param[in] indim The input dimension
  @param[in] outdim The output dimension
  */
  Model(std::size_t const indim, std::size_t const outdim);

  /**
  @param[in] pt Options for the model
  */
  Model(boost::property_tree::ptree const& pt);

  virtual ~Model() = default;

  /// Implement the nearest neighbor kernel
  /**
  The nearest neighbor kernel is a function \f$K:\mathbb{R}^{+} \mapsto \mathbb{R}{+}\f$ such that \f$K(0) = 1\f$ and \f$K\f$ monotonically (not necessarily stricly monotonically) decays so that \f$K(\delta) = 0\f$ if \f$\delta > 1\f$.

  Defaults to the hat kernel.
  @param[in] delta The input parameter (normalized distance between two points)
  \return The kernel evaluation
  */
  virtual double NearestNeighborKernel(double const delta) const;

  /// Implement the model operator \f$\mathcal{L}(u)\f$
  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is divided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  Eigen::VectorXd Operator(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const;

  /// Implement the identity operator \f$\mathcal{L}(u) = u\f$
  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is divided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation \f$\mathcal{L}(u) = u\f$
  */
  Eigen::VectorXd IdentityOperator(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const;

  /// Compute the \f$k^{th}\f$ derivative of the \f$i^{th}\f$ output with respect to the \f$j^{th}\f$ input \f$\frac{\partial^{k} u_i}{\partial x_j^{k}}\f$
  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is divided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  @param[in] i We are computing the derivative of the \f$i^{th}\f$ output
  @param[in] j We are computing the derivative with respect to the \f$j^{th}\f$ input
  @param[in] k We want the \f$k^{th}\f$ derivative
  \return The evaluation of \f$\frac{\partial^{k} u_i}{\partial x_j^{k}}\f$
  */
  double FunctionDerivative(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases, std::size_t const i, std::size_t const j, std::size_t const k) const;

  /// Compute the Jacobian of the model operator with respect to the basis coefficients using finite difference
  /**
  The \f$(i,j)\f$ entry of the returned matrix is
  \f{equation*}{
  \left. \frac{\partial (\mathcal{L}(u))_i }{\partial p_j} \right|_{x},
  \f}
  the derivative of the \f$i^{th}\f$ output with respect to the \f$j^{th}\f$ coefficient, evaluated at a point \f$x\f$.
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is divided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  @param[in] eval The evaluation of the operator at <tt>x</tt> with these <tt>coefficients</tt>, saves us from having to re-evaluate if we have this information handy. This defaults to empty; if its empty we compute this quantity
  @param[in] fdEps The step size for the finite difference approximation (defaults to <tt>1.0e-6</tt>)
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  Eigen::MatrixXd OperatorJacobianByFD(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases, Eigen::VectorXd const& eval = Eigen::VectorXd(), double const fdEps = 1.0e-6) const;

  /// Compute the Jacobian of the model operator with respect to the basis coefficients
  /**
  The \f$(i,j)\f$ entry of the returned matrix is
  \f{equation*}{
  \left. \frac{d (\mathcal{L}(u))_i }{d p_j} \right|_{x},
  \f}
  the derivative of the \f$i^{th}\f$ output with respect to the \f$j^{th}\f$ coefficient, evaluated at a point \f$x\f$.
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is divided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  Eigen::MatrixXd OperatorJacobian(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const;

  /// Compute the Jacobian (with respect to coefficients) of the identity operator given the bases
  /**
  The Jacobian of the identity is simply the basis evaluated at the given location
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] bases The basis functions for each output
  @param[in] numCoeffs The total number of basis functions, defaults to zero which means we need to compute this value by summing over the given basis functions
  \return The Jacobian of the identity operator
  */
  Eigen::MatrixXd IdentityOperatorJacobian(Eigen::VectorXd const& x, std::vector<std::shared_ptr<const BasisFunctions> > const& bases, std::size_t numCoeffs = 0) const;

  /// Compute the Hessian of the operator with respect to the basis coefficients
  /**
  The \f$(i,j)\f$ entry of the \f$k^{th}\f$ component in the vector of returned matrix is
  \f{equation*}{
  \left. \frac{\partial^2 (\mathcal{L}(u))_i }{\partial p_k \partial p_j} \right|_{x},
  \f}
  the derivative of the \f$i^{th}\f$ output with respect to the \f$j^{th}\f$ coefficient, evaluated at a point \f$x\f$.
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is divided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  std::vector<Eigen::MatrixXd> OperatorHessian(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const;

  /// Compute the Hessian of the operator with respect to the basis coefficients using finite difference
  /**
  The \f$(i,j)\f$ entry of the \f$k^{th}\f$ component in the vector of returned matrix is
  \f{equation*}{
  \left. \frac{\partial^2 (\mathcal{L}(u))_i }{\partial p_k \partial p_j} \right|_{x},
  \f}
  the derivative of the \f$i^{th}\f$ output with respect to the \f$j^{th}\f$ coefficient, evaluated at a point \f$x\f$.
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is divided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  @param[in] jacEval The evaluation of the operator Jacobian at <tt>x</tt> with these <tt>coefficients</tt>, saves us from having to re-evaluate if we have this information handy. This defaults to empty; if its empty we compute this quantity
  @param[in] fdEps The step size for the finite difference approximation (defaults to <tt>1.0e-6</tt>)
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  std::vector<Eigen::MatrixXd> OperatorHessianByFD(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases, Eigen::MatrixXd const& jacEval = Eigen::MatrixXd(), double const fdEps = 1.0e-6) const;

  /// Implement the right hand side function \f$f\f$
  /**
  @param[in] x The point \f$x \in \Omega \f$
  \return The evaluation of \f$f(x)\f$
  */
  Eigen::VectorXd RightHandSide(Eigen::VectorXd const& x) const;

  /// Is this a linear model?
  /**
  Defaults to <tt>false</tt>, but can be overriden by children. 
  \return <tt>true</tt>: The model is linear, <tt>false</tt>: The model is nonlinear
  */
  virtual bool IsLinear() const;

  /// The input dimension \f$d\f$
  const std::size_t inputDimension;

  /// The output dimension \f$m\f$
  const std::size_t outputDimension;

protected:

  /// Implement the right hand side function \f$f\f$
  /**
  @param[in] x The point \f$x \in \Omega \f$
  \return The evaluation of \f$f(x)\f$
  */
  virtual Eigen::VectorXd RightHandSideVectorImpl(Eigen::VectorXd const& x) const;

  /// Implement the action of the operator \f$\mathcal{L}(u)\f$
  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is divided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  virtual Eigen::VectorXd OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const = 0;

  /// Implement the Jacobian of action of the operator \f$\mathcal{L}(u)\f$
  /**
  The \f$(i,j)\f$ entry of the returned matrix is
  \f{equation*}{
  \left. \frac{\partial (\mathcal{L}(u))_i }{\partial p_j} \right|_{x},
  \f}
  the derivative of the \f$i^{th}\f$ output with respect to the \f$j^{th}\f$ coefficient, evaluated at a point \f$x\f$.
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is divided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  virtual Eigen::MatrixXd OperatorJacobianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const;

  /// Compute the Hessian of the operator with respect to the basis coefficients
  /**
  The \f$(i,j)\f$ entry of the \f$k^{th}\f$ component in the vector of returned matrix is
  \f{equation*}{
  \left. \frac{\partial^2 (\mathcal{L}(u))_i }{\partial p_k \partial p_j} \right|_{x},
  \f}
  the derivative of the \f$i^{th}\f$ output with respect to the \f$j^{th}\f$ coefficient, evaluated at a point \f$x\f$.
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is divided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  virtual std::vector<Eigen::MatrixXd> OperatorHessianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const;

  /// Implement the right hand side function \f$f\f$
  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] outind Return this component of the evaluation of \f$f\f$
  \return The component of \f$f(x)\f$ corresponding to <tt>outind</tt>
  */
  virtual double RightHandSideComponentImpl(Eigen::VectorXd const& x, std::size_t const outind) const;

private:

};

} // namespace clf

#endif
