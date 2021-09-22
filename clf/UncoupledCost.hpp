#ifndef UNCOUPLEDCOST_HPP_
#define UNCOUPLEDCOST_HPP_

#include <boost/property_tree/ptree.hpp>

#include "clf/CostFunction.hpp"

namespace clf {

/// Forward declaration of a support point
class SupportPoint;

/// Compute the uncoupled cost associated with the support point at \f$\hat{x}\f$
/**
Let \f$\Phi_{\hat{x}}(x)\f$ be the \f$m \times \tilde{q}\f$ matrix that defines the location function \f$\ell_{\hat{x}}(x) = \Phi_{\hat{x}}(x) p\f$ for \f$p \in \mathbb{R}^{\tilde{q}}\f$ (see clf::SupportPoint). The uncoupled cost function associated with the support point is
\f{equation*}{
p_i = \mbox{arg min}_{p \in \mathbb{R}^{\tilde{q}}} J(p) = \sum_{j=0}^{k_{nn}} \frac{m_i}{2} \| \mathcal{L}_{\hat{x}}(\Phi_{\hat{x}} (x_{I(\hat{x},j)}) p) - f(x_{I(\hat{x},j)}) \|^2 {K_i(x_i, x_{I(\hat{x},j)})} + \frac{a_i}{2} \|p\|^2,
\f}

<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"RegularizationParameter"   | <tt>double</tt> | <tt>0.0</tt> | The regularization parameter \f$a_i\f$. |
"UncoupledScale"   | <tt>double</tt> | <tt>1.0</tt> | The scale parameter \f$m_i\f$. |
*/
class UncoupledCost : public SparseCostFunction {
public:

  /**
  @param[in] point The point whose uncoupled cost we are computing
  @param[in] pt Options for the uncoupled cost function
  */
  UncoupledCost(std::shared_ptr<SupportPoint> const& point, boost::property_tree::ptree const& ptree);

  virtual ~UncoupledCost() = default;

  /// The parameter that scales the uncoupled cost
  /**
  \return The parameter that scales the uncoupled cost
  */
  double UncoupledScale() const;

  /// The parameter that scales the regularizing term
  /**
  \return The parameter that scales the regularizing term
  */
  double RegularizationScale() const;

  /// Compute the entries of the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the sub-cost function \f$f_i\f$ with respect to the input parameters \$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$.

  Since it is sparse we can store each entry as an <tt>Eigen::Triplet<double></tt>.

  @param[in] coefficients The coefficients for this support point
  @param[out] triplets The entries of the Jacobian matrix
  */
  //void JacobianTriplets(Eigen::VectorXd const& coefficients, std::vector<Eigen::Triplet<double> >& triplets) const;

  /// The point that is associated with this cost
  std::weak_ptr<const SupportPoint> point;

protected:

  /// Evaluate the \f$i^{th}\f$ penalty function \f$f_i(\beta)\f$
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The evaluation of the \f$i^{th}\f$ penalty function
  */
  virtual double PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const override;

  /// Evaluate each sub-cost function \f$f_i(\boldsymbol{\beta})\f$
  /**
  @param[in] beta The coefficients for this support point
  \return The \f$i^{th}\f$ entry is the \f$i^{th}\f$ sub-cost function \f$f_i(\boldsymbol{\beta})\f$. For the uncoupled cost, this is the difference between the model and right hand side associted with the \f$i^{th}\f$ nearest neighbor.
  */
  //Eigen::VectorXd CostImpl(Eigen::VectorXd const& beta) const;

  /// Compute the Jacobian matrix
  /**
  The Jacobian matrix is \f$\boldsymbol{J} \in \mathbb{R}^{m \times n}\f$. Each row is the gradient of the sub-cost function \f$f_i\f$ with respect to the input parameters \$\boldsymbol{\beta} \in \mathbb{R}^{n}\f$.

  @param[in] beta The coefficients for this support point
  @param[out] jac The Jacobian matrix
  */
  //void JacobianImpl(Eigen::VectorXd const& beta, Eigen::SparseMatrix<double>& jac) const;

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
};

} // namespace clf

#endif
