#ifndef COUPLEDCOST_HPP_
#define COUPLEDCOST_HPP_

#include <boost/property_tree/ptree.hpp>

#include <MUQ/Optimization/CostFunction.h>

namespace clf {

/// Forward declaration of a support point
class SupportPoint;

/// Compute the coupling cost associated with a support point
/**
The coupling cost function associated with support point \f$i\f$ is
\f{equation*}{
p^*_{I(i, 0)},\, p^*_{I(i, 1)},\, ...,\, p^*_{I(i, k_{nn})} = \mbox{arg min}_{p_{I(i, 0)},\, p_{I(i, 1)},\, ...,\, p_{I(i, k_{nn})}} J(p) = \sum_{j=1}^{k_{nn}} \frac{c_i}{2} \| \ell_i(x_{I(i,j)}, p_{I(i, 0)}) - \ell_{I(i,j)}(x_{I(i,j)}, p_{I(i, j)}) \|^2 {K_i(x_i, x_{I(i,j)})},
\f}

This class implements the cost of coupling between <em>two</em> neighboring points. Given the point \f$i\f$ and its \f$j^{th}\f$ nearest neighbor \f$I(i,j)\f$ this cost computes
\f{equation*}{
\frac{c_i}{2} \| \ell_i(x_{I(i,j)}, p_{I(i, 0)}) - \ell_{I(i,j)}(x_{I(i,j)}, p_{I(i, j)}) \|^2 {K_i(x_i, x_{I(i,j)})}.
\f}
Recall that \f$I(i,0) = i\f$ and
\f{equation*}{
\ell_i(x, p_{i}) = \left[ \begin{array}{c}
p_{i}^{(0)} \cdot \phi_i^{(0)}(x) \\
\vdots \\
p_{i}^{(q)} \cdot \phi_i^{(q)}(x)
\end{array} \right]
\f}
where \f$q\f$ is the number of outputs and \f$p_i^{(s)}\f$ is the coefficient segment that corresponds to the \f$s^{th}\f$ output.

<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"CoupledScale"   | <tt>double</tt> | <tt>1.0</tt> | The scale parameter \f$c_i\f$. |
*/
class CoupledCost : public muq::Optimization::CostFunction {
public:

  /**
  @param[in] point The point whose uncoupled cost we are computing
  @param[in] neigh The neighbor point that is coupled to the main point
  @param[in] pt Options for the uncoupled cost function
  */
  CoupledCost(std::shared_ptr<SupportPoint> const& point, std::shared_ptr<SupportPoint> const& neighbor, boost::property_tree::ptree const& pt);

  virtual ~CoupledCost() = default;

  /// Are these two points actually coupled?
  /**
  \return If this is <tt>false</tt> the cost is zero. Two points are <em>uncoupled</em> if they are not nearest neighbors or if they are the same point.
  */
  bool Coupled() const;

  /// Compute the Hessian of the cost function
  /**
  The Hessian of
  \f{equation*}{
  \frac{c_i}{2}  K_i(x_i, x_{I(i,j)}) \| \ell_i(x_{I(i,j)}, p_{I(i, 0)}) - \ell_{I(i,j)}(x_{I(i,j)}, p_{I(i, j)}) \|^2
  \f}
  with respect to \f$[p_i,\, p_{I(i,j)}]\f$ is
  \f{equation*}{
  c_i  K_i(x_i, x_{I(i,j)}) \left[ \begin{array}{cc}
  V_i^{\top} V_i & -V_i^{\top} V_{I(i,j)} \\
  -V_{I(i,j)}^{\top} V_i & V_{I(i,j)}^{\top} V_{I(i,j)}
  \end{array} \right],
  \f}
  where
  \f{equation*}{
  V_i = \nabla_{p_{i}} \ell_i(x, p_{i}) = \left[ \begin{array}{cccc}
  (\phi_i^{(0)}(x))^{\top} & 0 & ... & 0\\
  \vdots & \vdots & \vdots & \vdots \\
  0 & ... & 0 & (\phi_i^{(q)}(x))^{\top}
  \end{array} \right].
  \f}
  Note that
  \f{equation*}{
  V_i^{\top} V_i = \left[ \begin{array}{cccc}
  \phi_i^{(0)}(x) (\phi_i^{(0)}(x))^{\top} & 0 & ... & 0\\
  \vdots & \vdots & \vdots & \vdots \\
  0 & ... & 0 & \phi_i^{(q)}(x) (\phi_i^{(q)}(x))^{\top}
  \end{array} \right].
  \f}
  and
  \f{equation*}{
  V_i^{\top} V_{I(i,j)} = \left[ \begin{array}{cccc}
  \phi_i^{(0)}(x) (\phi_{I(i,j)}^{(0)}(x))^{\top} & 0 & ... & 0\\
  \vdots & \vdots & \vdots & \vdots \\
  0 & ... & 0 & \phi_i^{(q)}(x) (\phi_{I(i,j)}^{(q)}(x))^{\top}
  \end{array} \right].
  \f}
  are block-diagonal matrices.
  @param[out] ViVi Each entry is the block of the top left matrix in the Hessian corresponding to the \f$s^{th}\f$ output (blocks of \f$V_i^{\top}V_i\f$)
  @param[out] ViVj Each entry is the block of the top right matrix in the Hessian corresponding to the \f$s^{th}\f$ output (blocks of \f$V_i^{\top}V_{I(i,j)}\f$)---the transpose is the bottom left
  @param[out] VjVj Each entry is the block of the bottom right matrix in the Hessian corresponding to the \f$s^{th}\f$ output (blocks of \f$V_{I(i,j)}^{\top}V_{I(i,j)}\f$)
  */
  void Hessian(std::vector<Eigen::MatrixXd>& ViVi, std::vector<Eigen::MatrixXd>& ViVj, std::vector<Eigen::MatrixXd>& VjVj) const;

  /// The parameter that scales the coupling cost
  /**
  Defaults to \f$1.0\f$.
  */
  const double coupledScale;

protected:

  /// Compute the cost function
  /**
  @param[in] input There is only one input: the basis function coefficients
  */
  virtual double CostImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& input) override;

  /// Compute the cost function gradient
  /**
  The gradient of
  \f{equation*}{
  \frac{c_i}{2}  K_i(x_i, x_{I(i,j)}) \| \ell_i(x_{I(i,j)}, p_{I(i, 0)}) - \ell_{I(i,j)}(x_{I(i,j)}, p_{I(i, j)}) \|^2
  \f}
  with respect to \f$[p_i,\, p_{I(i,j)}]\f$ is
  \f{equation*}{
  c_i  K_i(x_i, x_{I(i,j)}) \left[ \begin{array}{c}
  V_i^{\top} ( \ell_i(x_{I(i,j)}, p_{I(i, 0)}) - \ell_{I(i,j)}(x_{I(i,j)}, p_{I(i, j)}) ) \\
  -V_{I(i,j)}^{\top} ( \ell_i(x_{I(i,j)}, p_{I(i, 0)}) - \ell_{I(i,j)}(x_{I(i,j)}, p_{I(i, j)}) )
  \end{array} \right],
  \f}
  where
  \f{equation*}{
  V_i = \nabla_{p_{i}} \ell_i(x, p_{i}) = \left[ \begin{array}{cccc}
  (\phi_i^{(0)}(x))^{\top} & 0 & ... & 0\\
  \vdots & \vdots & \vdots & \vdots \\
  0 & ... & 0 & (\phi_i^{(q)}(x))^{\top}
  \end{array} \right].
  \f}
  @param[in] inputDimWrt Since there is only one input, this should always be zero
  @param[in] input There is only one input: the basis function coefficients
  @param[in] sensitivity A scaling for the gradient
  */
  virtual void GradientImpl(unsigned int const inputDimWrt, muq::Modeling::ref_vector<Eigen::VectorXd> const& input, Eigen::VectorXd const& sensitivity) override;

private:

  /// Determine the local nearest neighbor index
  /**
  @param[in] point The point whose uncoupled cost we are computing
  @param[in] neigh The neighbor point that is coupled to the main point
  \return The local nearest neighbor index
  */
  static std::size_t LocalIndex(std::shared_ptr<SupportPoint> const& point, std::shared_ptr<SupportPoint> const& neighbor);

  /// The point where we are evaluating the coupling cost
  std::weak_ptr<SupportPoint> point;

  /// The nearest neighbor that (may be) coupled with the main support point
  std::weak_ptr<SupportPoint> neighbor;

  const std::vector<Eigen::VectorXd> pointBasisEvals;

  /// The neighbor's basis functions evaluated at the neighbor support point
  const std::vector<Eigen::VectorXd> neighborBasisEvals;

  /// The local nearest neighbor index
  /**
  The neighbor point is the \f$j^{th}\f$ nearest neighbor of the main support point.

  If this is an invalid index, the cost is zero. Two points are <em>uncoupled</em> if they are not nearest neighbors or if they are the same point. Invalid indices are stored as the maximum possible integer.
  */
  const std::size_t localNeighborInd;

};

} // namespace clf

#endif
