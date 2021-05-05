#ifndef UNCOUPLEDCOST_HPP_
#define UNCOUPLEDCOST_HPP_

#include <boost/property_tree/ptree.hpp>

#include <MUQ/Optimization/CostFunction.h>

namespace clf {

/// Forward declaration of a support point
class SupportPoint;

/// Compute the uncoupled cost associated with a support point
/**
The uncoupled cost function associated with support point \f$i\f$ is
\f{equation*}{
p_i = \mbox{arg min}_{p \in \mathbb{R}^{\bar{q}_i}} J(p) = \sum_{j=1}^{k_{nn}} \frac{m_i}{2} \| \mathcal{L}_i(\hat{u}(x_{I(i,j)}, p)) - f_i(x_{I(i,j)}) \|^2 {K_i(x_i, x_{I(i,j)})} + \frac{a_i}{2} \|p\|^2,
\f}

<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"RegularizationParameter"   | <tt>double</tt> | <tt>0.0</tt> | The regularization parameter \f$a_i\f$. |
"UncoupledScale"   | <tt>double</tt> | <tt>1.0</tt> | The scale parameter \f$m_i\f$. |
*/
class UncoupledCost : public muq::Optimization::CostFunction {
public:

  /**
  @param[in] point The point whose uncoupled cost we are computing
  @param[in] pt Options for the uncoupled cost function
  */
  UncoupledCost(std::shared_ptr<SupportPoint> const& point, boost::property_tree::ptree const& ptree);

  virtual ~UncoupledCost() = default;

  /// Compute the cost given a segment of the gobal coefficient vector
  /**
  @param[in] coefficients The coefficients for the basis functions
  \return The uncoupled cost
  */
  double Cost(Eigen::VectorXd const& coefficients) const;

  /// Compute the Hessian of the cost function
  /**
  @param[in] coefficients The coefficients for the basis functions
  @param[in] gaussNewtonHessian <tt>true</tt>: compute the Gauss-Newton Hessian, <tt>false</tt>: compute the Hessian
  \return The Hessian matrix (or the Gauss-Newton Hessian matrix)
  */
  Eigen::MatrixXd Hessian(Eigen::VectorXd const& coefficients, bool const gaussNewtonHessian) const;

  /// The point that is associated with this cost
  std::weak_ptr<const SupportPoint> point;

  /// The parameter that scales the uncoupled cost
  /**
  Defaults to \f$1.0\f$.
  */
  double uncoupledScale;

  /// The parameter that scales the regularizing term
  /**
  Defaults to \f$0.0\f$.
  */
  double regularizationScale;

protected:

  /// Compute the cost function
  /**
  @param[in] input There is only one input: the basis function coefficients
  */
  virtual double CostImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& input) override;

  /// Compute the cost function gradient
  /**
  @param[in] inputDimWrt Since there is only one input, this should always be zero
  @param[in] input There is only one input: the basis function coefficients
  @param[in] sensitivity A scaling for the gradient
  */
  virtual void GradientImpl(unsigned int const inputDimWrt, muq::Modeling::ref_vector<Eigen::VectorXd> const& input, Eigen::VectorXd const& sensitivity) override;

private:
};

} // namespace clf

#endif
