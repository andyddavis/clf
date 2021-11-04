#ifndef LINEARMODEL_HPP_
#define LINEARMODEL_HPP_

#include "clf/Model.hpp"

namespace clf {

/// Implement a linear model with the form \f$\mathcal{L}(u(x)) \approx \mathcal{L}(\Phi_{\hat{x}}(x) p) = L_{\hat{x}}(x) p\f$, where \f$L_{\hat{x}} \in \mathbb{R}^{m \times \tilde{q}}\f$ and \f$p \in \mathbb{R}^{\tilde{q}}\f$ are the coefficents associated with the clf::SupportPoint at \f$\hat{x}\f$.
class LinearModel : public Model {
public:
  /// Construct a linear model given the in/output dimension
  /**
  @param[in] indim The input dimension
  @param[in] outdim The output dimension
  */
  LinearModel(std::size_t const indim, std::size_t const outdim);

  /// Construct a linear model, by default use the identity model \f$\mathcal{L}(u) = u\f$
  /**
  @param[in] pt Options for the model
  */
  LinearModel(boost::property_tree::ptree const& pt);

  virtual ~LinearModel() = default;

  /// The matrix \f$L_{\hat{x}}\f$ that defines the linear model
  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] bases The basis functions for each output
  \return The matrix \f$L_{\hat{x}}(x)\f$ that defines the linear model
   */
  virtual Eigen::MatrixXd ModelMatrix(Eigen::VectorXd const& x, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const;

  /// Is this a linear model?
  /**
  Returns <tt>true</tt> since clearly this model must be linear.
  \return <tt>true</tt>: The model is linear
  */
  virtual bool IsLinear() const final override;

protected:

  /// Implement the linear model
  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is divided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  virtual Eigen::VectorXd OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const final override;

  /// Implement the linear model gradient
  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is divided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of the gradient with respect to the coefficeints
  */
  virtual Eigen::MatrixXd OperatorJacobianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const final override;

  /// Implement the linear model Hessian
  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is divided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of the Hessian with respect to the coefficients
  */
  virtual std::vector<Eigen::MatrixXd> OperatorHessianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const final override;

private:
};

} // namespace clf

#endif
