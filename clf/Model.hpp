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
where \f$u:\Omega \mapsto \mathbb{R}^{m}\f$ (and \f$\Omega \subseteq \mathbb{R}^{d}\f$) is the function that we are trying to represent with local functions. This class allows the user to implement \f$\mathcal{L}\f$ and/or \f$f\f$. By default, we assume \f$\mathcal{L}\f$ is the identity but the user <em>must</em> implement \f$f\f$.

<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"InputDimension"   | <tt>std::size_t</tt> | <tt>1</tt> | The input dimension \f$d\f$. |
"OutputDimension"   | <tt>std::size_t</tt> | <tt>1</tt> | The output dimension \f$m\f$. |
*/
class Model {
public:

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
  @param[in] coefficients The coefficients for each basis---this vector is devided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  virtual Eigen::VectorXd Operator(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const;

  /// Implement the identity operator \f$\mathcal{L}(u)\f$
  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is devided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  virtual Eigen::VectorXd IdentityOperator(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const;

  /// Compute the Jacobian of the model operator with respect to the basis coefficients using finite difference
  /**
  The \f$(i,j)\f$ entry of the returned matrix is
  \f{equation*}{
  \left. \frac{d (\mathcal{L}(u))_i }{d p_j} \right|_{x},
  \f}
  the derivative of the \f$i^{th}\f$ output with respect to the \f$j^{th}\f$ coefficient, evaluated at a point \f$x\f$.
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is devided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  virtual Eigen::MatrixXd OperatorJacobianByFD(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const;

  /// Compute the Jacobian of the model operator with respect to the basis coefficients
  /**
  The \f$(i,j)\f$ entry of the returned matrix is
  \f{equation*}{
  \left. \frac{d (\mathcal{L}(u))_i }{d p_j} \right|_{x},
  \f}
  the derivative of the \f$i^{th}\f$ output with respect to the \f$j^{th}\f$ coefficient, evaluated at a point \f$x\f$.
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is devided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  virtual Eigen::MatrixXd OperatorJacobian(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const;

  /// Implement the right hand side function \f$f\f$
  /**
  @param[in] x The point \f$x \in \Omega \f$
  \return The evaluation of \f$f(x)\f$
  */
  Eigen::VectorXd RightHandSide(Eigen::VectorXd const& x) const;

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
  @param[in] coefficients The coefficients for each basis---this vector is devided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  virtual Eigen::VectorXd OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const;

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
