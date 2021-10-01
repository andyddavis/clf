#ifndef COUPLEDCOST_HPP_
#define COUPLEDCOST_HPP_

#include <boost/property_tree/ptree.hpp>

#include "clf/CostFunction.hpp"

namespace clf {

/// Forward declaration of a support point
class SupportPoint;

/// Compute the coupling cost associated with a support point \f$\hat{x}\f$ and its \f$i^{th}\f$ nearest neighbor \f$x_i\f$
/**
Let \f$\Phi_{\hat{x}}(x)\f$ be the \f$m \times \tilde{q}_{\hat{x}}\f$ matrix that defines the local function \f$\ell_{\hat{x}}(x) = \Phi_{\hat{x}}(x) p\f$ for \f$p \in \mathbb{R}^{\tilde{q}_{\hat{x}}}\f$ (see clf::SupportPoint). Similarly, let \f$\Phi_{x_i}(x)\f$ be the \f$m \times \tilde{q}_i\f$ matrix that defines the local function \f$\ell_{x_i}(x) = \Phi_{x_i}(x) s\f$ for \f$s \in \mathbb{R}^{\tilde{q}_i}\f$ associated with the neighbor point. The coupling cost function between support point \f$\hat{x}\f$ and its \f$i^{th}\f$ nearest neighbor \f$x_i\f$ is 
\f{equation*}{
J(p, s) = \frac{c_i}{2} \| \Phi_{\hat{x}}(x_i) p - \Phi_{x_i}(x_i) s \|^2 K_i(\hat{x}, x_{i}).
\f}
The input coefficeints are \f$[p, s]^{\top} \in \mathbb{R}^{\tilde{q}_{\hat{x}}+\tilde{q}_i}\f$.

<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"CoupledScale"   | <tt>double</tt> | <tt>1.0</tt> | The scale parameter \f$c_i\f$. |
*/
class CoupledCost : public DenseCostFunction {
public:
  /**
  @param[in] point The point use uncoupled cost we are computing
  @param[in] neighbor The neighbor point that is coupled to the main point
  @param[in] pt Options for the uncoupled cost function
  */
  CoupledCost(std::shared_ptr<SupportPoint> const& point, std::shared_ptr<SupportPoint> const& neighbor, boost::property_tree::ptree const& pt);

  virtual ~CoupledCost() = default;

  /// Are these two points actually coupled?
  /**
  \return If this is <tt>false</tt> the cost is zero. Two points are <em>uncoupled</em> if they are not nearest neighbors or if they are the same point.
  */
  bool Coupled() const;

  /// Get a pointer to the clf::SupportPoint associated with \f$\hat{x}\f$
  /**
  \return A pointer to the clf::SupportPoint associated with \f$\hat{x}\f$
  */
  std::shared_ptr<const SupportPoint> GetPoint() const;

  /// Get a pointer to the clf::SupportPoint associated with \f$x_i\f$
  /**
  \return A pointer to the clf::SupportPoint associated with \f$x_i\f$
  */
  std::shared_ptr<const SupportPoint> GetNeighbor() const;

  /// The parameter that scales the coupled cost (the parameter \f$c_i\f$)
  /**
  \return The parameter that scales the coupled cost
  */
  double CoupledScale() const;

  /// Evaluate the penalty function given the coefficients \f$[p, s]^{\top} \in \mathbb{R}^{\tilde{q}_{\hat{x}}+\tilde{q}_i}\f$
  /**
  The penalty function is 
  \f{equation*}{
  \sqrt{\frac{c_i}{2} K_i(\hat{x}, x_{i})} \| \Phi_{\hat{x}}(x_i) p - \Phi_{x_i}(x_i) s \|.
  \f}
  @param[in] coeffPoint The coefficients \f$p \in \mathbb{R}^{\tilde{q}_{\hat{x}}}\f$
  @param[in] coeffNeigh The coefficients \f$s \in \mathbb{R}^{\tilde{q}_i}\f$
  \return The evaluation of the penalty function
  */
  double PenaltyFunction(Eigen::VectorXd const& coeffPoint, Eigen::VectorXd const& coeffNeigh) const;

  /// Evaluate the gradient of the penalty function given the coefficients \f$[p, s]^{\top} \in \mathbb{R}^{\tilde{q}_{\hat{x}}+\tilde{q}_i}\f$
  /**
  The gradient of the penalty function is 
  \f{equation*}{
  \frac{ \sqrt{\frac{c_i}{2} K_i(\hat{x}, x_{i})} }{ \| \Phi_{\hat{x}}(x_i) p - \Phi_{x_i}(x_i) s \| } \left[ 
  \begin{array}{c}
  \Phi_{\hat{x}}(x_i)^{\top} (\Phi_{\hat{x}}(x_i) p - \Phi_{x_i}(x_i) s) \\
  - \Phi_{x_i}(x_i)^{\top} (\Phi_{\hat{x}}(x_i) p - \Phi_{x_i}(x_i) s)
  \end{array}
  \right].
  \f}
  Note that the gradient is zero if \f$\| \Phi_{\hat{x}}(x_i) p - \Phi_{x_i}(x_i) s \| = 0\f$.
  @param[in] coeffPoint The coefficients \f$p \in \mathbb{R}^{\tilde{q}_{\hat{x}}}\f$
  @param[in] coeffNeigh The coefficients \f$s \in \mathbb{R}^{\tilde{q}_i}\f$
  \return The gradient of the penalty function
  */
  Eigen::VectorXd PenaltyFunctionGradient(Eigen::VectorXd const& coeffPoint, Eigen::VectorXd const& coeffNeigh) const;

  /// Is this a quadratic cost function?
  /**
  The coupling cost is always quadratic.
  \return <tt>true</tt>: The cost function is quadratic
  */
  virtual bool IsQuadratic() const override;

protected:

  /// Evaluate the penalty function given the coefficients \f$\beta = [p, s]^{\top} \in \mathbb{R}^{\tilde{q}_{\hat{x}}+\tilde{q}_i}\f$
  /**
  The penalty function is 
  \f{equation*}{
  \sqrt{\frac{c_i}{2} K_i(\hat{x}, x_{i})} \| \Phi_{\hat{x}}(x_i) p - \Phi_{x_i}(x_i) s \|.
  \f}
  The input coefficeints are \f$\beta = [p, s]^{\top} \in \mathbb{R}^{\tilde{q}_{\hat{x}}+\tilde{q}_i}\f$.
  @param[in] ind The index of the penalty function (must be \f$0\f$ since there is only one penalty function)
  @param[in] beta The input parameters \f$\beta = [p, s]^{\top} \in \mathbb{R}^{\tilde{q}_{\hat{x}}+\tilde{q}_i}\f$
  \return The evaluation of the penalty function
  */
  virtual double PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const override;

  /// Evaluate the gradient of the penalty function given the coefficients \f$\beta = [p, s]^{\top} \in \mathbb{R}^{\tilde{q}_{\hat{x}}+\tilde{q}_i}\f$
  /**
  The gradient of the penalty function is 
  \f{equation*}{
  \frac{ \sqrt{\frac{c_i}{2} K_i(\hat{x}, x_{i})} }{ \| \Phi_{\hat{x}}(x_i) p - \Phi_{x_i}(x_i) s \| } \left[ 
  \begin{array}{c}
  \Phi_{\hat{x}}(x_i)^{\top} (\Phi_{\hat{x}}(x_i) p - \Phi_{x_i}(x_i) s) \\
  - \Phi_{x_i}(x_i)^{\top} (\Phi_{\hat{x}}(x_i) p - \Phi_{x_i}(x_i) s)
  \end{array}
  \right].
  \f}
  The input coefficeints are \f$\beta = [p, s]^{\top} \in \mathbb{R}^{\tilde{q}_{\hat{x}}+\tilde{q}_i}\f$. Note that the gradient is zero if \f$\| \Phi_{\hat{x}}(x_i) p - \Phi_{x_i}(x_i) s \| = 0\f$.
  @param[in] ind The index of the penalty function (must be \f$0\f$ since there is only one penalty function)
  @param[in] beta The input parameters \f$\beta = [p, s]^{\top} \in \mathbb{R}^{\tilde{q}_{\hat{x}}+\tilde{q}_i}\f$
  \return The gradient of the penalty function
  */
  virtual Eigen::VectorXd PenaltyFunctionGradientImpl(std::size_t const ind, Eigen::VectorXd const& beta) const override;

private:

  /// Determine the local nearest neighbor index
  /**
  If \f$x_i\f$ is one of the \f$k_{nn}\f$ nearest neighbors of \f$\hat{x}\f$ then this function returns the local index \f$\hat{i} \in [1, k_{nn}]\f$ that indicates which nearest neighbor. However, if \f$x_i\f$ is not one of the nearest neighbors or it is equal to \f$\hat{x}\f$, then this function returns <tt>std::numeric_limits<std::size_t>::max()</tt>.
  @param[in] point The point whose uncoupled cost we are computing
  @param[in] neigh The neighbor point that is coupled to the main point
  \return The local nearest neighbor index (or <tt>std::numeric_limits<std::size_t>::max()</tt>)
  */
  static std::size_t LocalIndex(std::shared_ptr<SupportPoint> const& point, std::shared_ptr<SupportPoint> const& neighbor);

  /// The point \f$\hat{x}\f$ where we are evaluating the coupling cost
  std::weak_ptr<const SupportPoint> point;

  /// The nearest neighbor \f$x_i\f$ that (may be) coupled with the main support point
  /**
  If this point is not one of the \f$k_{nn}\f$ nearest neighbors of clf::CoupledCost::point or if it equal to clf::CoupledCost::point, then the coupling cost is zero.
  */
  std::weak_ptr<const SupportPoint> neighbor;

  /// The point's basis functions evaluated at the neighbor support point
  /**
  The matrix \f$\Phi_{\hat{x}}(x_i)\f$ is defined by \f$m\f$ vectors \f$\phi_{\hat{x}}^{(j)}(x_i)\f$ (see clf::SupportPoint). This is a vector of length \f$m\f$ such that the \f$j^{th}\f$ entry is \f$\phi_{\hat{x}}^{(j)}(x_i)\f$.
  */
  const std::vector<Eigen::VectorXd> pointBasisEvals;

  /// The neighbor's basis functions evaluated at the neighbor support point
  /**
  The matrix \f$\Phi_{x_i}(x_i)\f$ is defined by \f$m\f$ vectors \f$\phi_{x_i}^{(j)}(x_i)\f$ (see clf::SupportPoint). This is a vector of length \f$m\f$ such that the \f$j^{th}\f$ entry is \f$\phi_{x_i}^{(j)}(x_i)\f$.
  */
  const std::vector<Eigen::VectorXd> neighborBasisEvals;

  /// The local nearest neighbor index
  /**
  The neighbor point is the \f$j^{th}\f$ nearest neighbor of the support point associated with \f$\hat{x}\f$.

  If this is an invalid index, the cost is zero. Two points are <em>uncoupled</em> if they are not nearest neighbors or if they are the same point. Invalid indices are stored as the maximum possible integer (<tt>std::numeric_limits<std::size_t>::max()</tt>).
  */
  const std::size_t localNeighborInd;

  /// The (precomputed) scale for the cost function
  /**
  The cost function for the Levenberg Marquardt algorithm takes the form \f$\sum_{i=1}^{n} f_i^2\f$. Therefore, the constant \f$c_i K_i(\hat{x}, x_i)\f$ actually corresponds to multiplying by \f$\sqrt{c_i K_i(\hat{x}, x_i)/2}\f$
  */
  const double scale;
};

} // namespace clf

#endif
