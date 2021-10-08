#ifndef COLLOCATIONCOST_HPP_
#define COLLOCATIONCOST_HPP_

#include "clf/CollocationPointCloud.hpp"
#include "clf/SparseCostFunction.hpp"

namespace clf {

/// The cost function to minimize the expected least squares error with respect to a given measure
/**
Let \f$\Phi_{\hat{x}}(x)\f$ be the \f$m \times \tilde{q}\f$ matrix that defines the local function \f$\ell_{\hat{x}}(x) = \Phi_{\hat{x}}(x) p_{\hat{x}}\f$ for \f$p_{\hat{x}} \in \mathbb{R}^{\tilde{q}(\hat{x})}\f$ (see clf::SupportPoint). Here, \f$\hat{x}\f$ is the nearest support point (see clf::SupportPoint). Define the approximation \f$\hat{u}(x) = \Phi_{\hat{x}(x)}(x) \hat{p}_{\hat{x}(x)}\f$, where \f$\hat{x}(x)\f$ is the nearest support point to \f$x\f$. 

Suppose we have \f$n\f$ support points \f$\{x_i\}_{i=1}^{n}\f$, we want to find \f$\hat{w} \in \mathbb{R}^{n \times d}\f$ such that
\f{equation*}{
\hat{w} = \mbox{arg min}_{w \in \mathbb{R}^{n \times d}} \int_{\Omega} \frac{1}{2} \| \mathcal{L}(\Phi_{x_{I(x)}}(x) \hat{p}_{I(x)}) - f \|^2 \pi \, dx,
\f}
where \f$I(x)\f$ is the index of the closest support point to \f$x\f$ and
\f{equation*}{
\hat{p} = \mbox{arg min}_{[n_1, ..., p_n] \in \mathbb{R}^{\tilde{q}_1} \times ... \times \mathbb{R}^{\tilde{q}_n}} \sum_{i=1}^{n} \underbrace{\sum_{j=1}^{k_i} \frac{1}{2} \|\Phi_{x_{i}}(x_j) \hat{p}_{i} - w_{j,:} \|^2 K_{x_i}(x_i, x_j) + \frac{a_i}{2} \| p_i \|}_{\text{uncoupled cost}} + \underbrace{\sum_{j=1}^{k_i} \frac{c_i}{2} \|\Phi_{x_{i}}(x_j) \hat{p}_{i}  - \Phi_{x_{j}}(x_j) \hat{p}_{j} \|^2 K_{x_i}(x_i, x_j)}_{\text{Coupled cost}}.
\f}
The uncoupled cost and coupled cost are computed in clf::UncoupledCost and clf::CoupledCost, respectively.
*/
class CollocationCost : public SparseCostFunction {
public:

  /**
  @param[in] collocationCloud The collocation cloud that we will use to compute this cost
  */
  CollocationCost(std::shared_ptr<CollocationPointCloud> const& colocationCloud);

  virtual ~CollocationCost() = default;

  /// Compute the optimal coefficients for the local polynomials associated with each support point
  /**
  @param[in] data Each column is the function we are tryng to approximate evaluated at the support point
  */
  void ComputeOptimalCoefficients(Eigen::MatrixXd const& data) const;

  /// Implement the sub-cost functions
  /**
  @param[in] data Each column is the function we are tryng to approximate evaluated at the support point
  */
  Eigen::VectorXd ComputeCost(Eigen::MatrixXd const& data) const;

  /// Is this a quadratic cost function?
  /**
  This colocation cost is quadratic if all of the models are linear
  \return <tt>true</tt>: The cost function is quadratic, <tt>false</tt>: The cost function is not quadratic
  */
  virtual bool IsQuadratic() const override;

private:

  /// Evaluate the \f$i^{th}\f$ penalty function \f$f_i(\beta)\f$
  /**
  @param[in] ind The index of the penalty function
  @param[in] beta The input parameter
  \return The evaluation of the \f$i^{th}\f$ penalty function
  */
  virtual Eigen::VectorXd PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const override;

  /// Implement the sub-cost functions
  /**
  @param[in] data When mapped into a matrix, each column is the function we are tryng to approximate evaluated at the support point
  */
  Eigen::VectorXd CostImpl(Eigen::VectorXd const& data) const;

  /**
  @param[in] data When mapped into a matrix, each column is the function we are tryng to approximate evaluated at the support point
  */
  void JacobianImpl(Eigen::VectorXd const& data, Eigen::SparseMatrix<double>& jac) const;

  /// The collocation cloud that we will use to comptue this cost
  std::shared_ptr<CollocationPointCloud> collocationCloud;
};

} // namespace clf

#endif
