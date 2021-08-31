#ifndef COUPLEDCOST_HPP_
#define COUPLEDCOST_HPP_

#include <boost/property_tree/ptree.hpp>

#include "clf/CostFunction.hpp"

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
class CoupledCost : public SparseCostFunction {
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

  /// Get a pointer to the point whose coupling cost we are computing
  /**
  \return A pointer to the point whose coupling cost we are computing
  */
  std::shared_ptr<const SupportPoint> GetPoint() const;

  /// Get a pointer to the neighbor
  /**
  \return A pointer to the neighbor
  */
  std::shared_ptr<const SupportPoint> GetNeighbor() const;

  /// Compute the coupled cost
  /**
  @param[in] coeffPoint The coefficients for the points whose coupling cost we are computing
  @param[in] coeffNeigh The coefficients for the neighbor
  \return Each entry is an evaluation of a sub-cost function
  */
  Eigen::VectorXd ComputeCost(Eigen::VectorXd const& coeffPoint, Eigen::VectorXd const& coeffNeigh) const;

  /// Compute the entries of the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the sub-cost function \f$f_i\f$ with respect to the input parameters \$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$.

  Since it is sparse we can store each entry as an <tt>Eigen::Triplet<double></tt>.

  @param[out] triplets The entries of the Jacobian matrix
  */
  void JacobianTriplets(std::vector<Eigen::Triplet<double> >& triplets) const;

protected:

  /// Evaluate each sub-cost function \f$f_i(\boldsymbol{\beta})\f$
  /**
  @param[in] beta The coefficients for this support point and its \f$i^{th}\f$ nearest neighbor
  \return The \f$i^{th}\f$ entry is the \f$i^{th}\f$ sub-cost function \f$f_i(\boldsymbol{\beta})\f$. For the coupled cost, this is the difference between this local function at this support point and its \f$i^{th}\f$ nearest neighbor evaluated at the neighbor.
  */
  virtual Eigen::VectorXd CostImpl(Eigen::VectorXd const& beta) const override;

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the sub-cost function \f$f_i\f$ with respect to the input parameters \$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$.

  @param[in] beta The coefficients for this support point and its \f$i^{th}\f$ nearest neighbor
  @param[out] jac The Jacobian matrix
  */
  virtual void JacobianImpl(Eigen::VectorXd const& beta, Eigen::SparseMatrix<double>& jac) const override;

private:

  /// Determine the local nearest neighbor index
  /**
  @param[in] point The point whose uncoupled cost we are computing
  @param[in] neigh The neighbor point that is coupled to the main point
  \return The local nearest neighbor index
  */
  static std::size_t LocalIndex(std::shared_ptr<SupportPoint> const& point, std::shared_ptr<SupportPoint> const& neighbor);

  /// The point where we are evaluating the coupling cost
  std::weak_ptr<const SupportPoint> point;

  /// The nearest neighbor that (may be) coupled with the main support point
  std::weak_ptr<const SupportPoint> neighbor;

  /// The point's basis functions evaluated at the neighbor support point
  const std::vector<Eigen::VectorXd> pointBasisEvals;

  /// The neighbor's basis functions evaluated at the neighbor support point
  const std::vector<Eigen::VectorXd> neighborBasisEvals;

  /// The local nearest neighbor index
  /**
  The neighbor point is the \f$j^{th}\f$ nearest neighbor of the main support point.

  If this is an invalid index, the cost is zero. Two points are <em>uncoupled</em> if they are not nearest neighbors or if they are the same point. Invalid indices are stored as the maximum possible integer (<tt>std::numeric_limits<std::size_t>::max()</tt>).
  */
  const std::size_t localNeighborInd;

    /// The (precomputed) scale for the cost function
  /**
  The cost function for the Levenberg Marquardt algorithm takes the form \f$\sum_{i=1}^{n} f_i^2\f$. Therefore, the constant \f$c_i K_i(x_i, x_{I(i,j)})\f$ actually corresponds to multiplying by \f$\sqrt{c_i K_i(x_i, x_{I(i,j)})}\f$
  */
  const double scale;
};

} // namespace clf

#endif
