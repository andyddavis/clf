#ifndef UNCOUPLEDCOST_HPP_
#define UNCOUPLEDCOST_HPP_

#include <boost/property_tree/ptree.hpp>
#include <boost/optional.hpp>

#include "clf/CostFunction.hpp"

namespace clf {

/// Forward declaration of a support point
class SupportPoint;

/// Compute the uncoupled cost associated with the support point at \f$\hat{x}\f$
/**
   Let \f$\Phi_{\hat{x}}(x)\f$ be the \f$m \times \tilde{q}\f$ matrix that defines the local function \f$\ell_{\hat{x}}(x) = \Phi_{\hat{x}}(x) p\f$ for \f$p \in \mathbb{R}^{\tilde{q}}\f$ (see clf::SupportPoint). The uncoupled cost function associated with the support point is
\f{equation*}{
p_{\hat{x}} = \mbox{arg min}_{p \in \mathbb{R}^{\tilde{q}}} J(p) = \sum_{j=0}^{k_{nn}} \frac{m}{2} \| \mathcal{L}_{\hat{x}}(\Phi_{\hat{x}} (x_{I(\hat{x},j)}) p) - f(x_{I(\hat{x},j)}) \|^2 {K_{\hat{x}}(\hat{x}, x_{I(\hat{x},j)})} + \frac{a}{2} \|p\|^2,
\f}
where \f$I(\hat{x}, j)\f$ is the \f$j^{th}\f$ closest point to the support point \f$\hat{x}\f$.

<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"RegularizationParameter"   | <tt>double</tt> | <tt>0.0</tt> | The regularization parameter \f$a\f$. |
"UncoupledScale"   | <tt>double</tt> | <tt>1.0</tt> | The scale parameter \f$m\f$. |
*/
class UncoupledCost : public DenseCostFunction {
public:

  /// Construct the clf::UncoupledCost
  /**
  @param[in] point The point whose uncoupled cost we are computing
  @param[in] pt Options for the uncoupled cost function
  */
  UncoupledCost(std::shared_ptr<SupportPoint> const& point, boost::property_tree::ptree const& ptree);

  virtual ~UncoupledCost() = default;

  /// The parameter that scales the uncoupled cost (the parameter \f$m\f$)
  /**
  \return The parameter that scales the uncoupled cost
  */
  double UncoupledScale() const;

  /// The parameter that scales the regularizing term (the parameter \f$a\f$)
  /**
  \return The parameter that scales the regularizing term
  */
  double RegularizationScale() const;

  /// Evaluate the forcing function \f$f\f$ at the \f$i^{th}\f$ support point \f$x_i\f$
  /**
  @param[in] pnt We need the forcing function \f$f\f$ evaluated at this support point 
  */
  Eigen::VectorXd EvaluateForcingFunction(std::shared_ptr<SupportPoint> const& pnt) const;  

  /// Set clf::UncoupledCost::forcing so that we use precoupled values the forcing function \f$f(x_i)\f$ at each support point \f$x_i\f$.
  /**
  @param[in] force The \f$i^{th}\f$ column is the forcing function evaluated at the \f$i^{th}\f$ support point \f$f(x_i)\f$
  */
  void SetForcingEvaluations(Eigen::MatrixXd const& force);

  /// Set clf::UncoupledCost::forcing equal to <tt>boost::none</tt>
  /**
  This means that those evaluation for the forcing function are no longer used to compute the uncoupled cost.
  */
  void UnsetForcingEvaluations();

  /// Is this a quadratic cost function?
  /**
  The uncoupled cost is quadratic if all of the functions are linear
  \return <tt>true</tt>: The cost function is quadratic, <tt>false</tt>: The cost function is not quadratic
  */
  virtual bool IsQuadratic() const override;

  /// The point \f$\hat{x}\f$ that is associated with this cost
  std::weak_ptr<const SupportPoint> point;

protected:

  /// Evaluate the \f$i^{th}\f$ penalty function 
  /**
  For \f$i \in [0, k_{nn}]\f$ where \f$k_{nn}\f$ is the number of nearest neighbors the \f$i^{th}\f$ penalty function is
  \f{equation*}{
    \sqrt{\frac{m}{2} K_i(x_i, x_{I(\hat{x},j)})} \| \mathcal{L}_{\hat{x}}(\Phi_{\hat{x}} (x_{I(\hat{x},j)}) p) - f(x_{I(\hat{x},j)}) \|
  \f}
  and for \f$i=k_{nn}+1\f$ (only if \f$a>0\f$) the final penalty function is
  \f{equation*}{
    \sqrt{\frac{a}{2}} \|p\|.
  \f}

  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The evaluation of the \f$i^{th}\f$ penalty function
  */
  virtual double PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const override;

  /// Evaluate the gradient of the \f$i^{th}\f$ penalty function
  /**
  For \f$i \in [0, k_{nn}]\f$ where \f$k_{nn}\f$ is the number of nearest neighbors the \f$i^{th}\f$ penalty function gradient is
  \f{equation*}{
    \sqrt{\frac{m}{2} K_i(x_i, x_{I(\hat{x},j)})} \frac{(\nabla_{p} \mathcal{L}_{\hat{x}}(\Phi_{\hat{x}} (x_{I(\hat{x},j)}) p))^{\top} ( \mathcal{L}_{\hat{x}}(\Phi_{\hat{x}} (x_{I(\hat{x},j)}) p) - f(x_{I(\hat{x},j)}) ) }{ \| \mathcal{L}_{\hat{x}}(\Phi_{\hat{x}} (x_{I(\hat{x},j)}) p) - f(x_{I(\hat{x},j)}) \| }
  \f}
  and for \f$i=k_{nn}+1\f$ (only if \f$a>0\f$) the final penalty function gradient is
  \f{equation*}{
    \sqrt{\frac{a}{2}} \frac{p}{\|p\|}.
  \f}

  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The gradient of the \f$i^{th}\f$ penalty function
  */
  virtual Eigen::VectorXd PenaltyFunctionGradientImpl(std::size_t const ind, Eigen::VectorXd const& beta) const override;
private:

  /// The parameter that scales the uncoupled cost
  /**
  Defaults to \f$1.0\f$.

  We actually store the square root of the uncoupled scale to save a little bit of computation.
  */
  double uncoupledScale;

  /// The parameter that scales the regularizing term
  /**
  Defaults to \f$0.0\f$.

  We actually store the square root of the uncoupled scale to save a little bit of computation.
  */
  double regularizationScale;

  /// The \f$i^{th}\f$ column is the forcing function evaluated at the \f$i^{th}\f$ support point \f$f(x_i)\f$
  /**
  If this optional parameter is set, use these values for the forcing function \f$f(x_i)\f$ evaluated at each support point. Otherwise, use the implementation in clf::Model::RightHandSide.
  */
  boost::optional<Eigen::MatrixXd const&> forcing;
};

} // namespace clf

#endif
