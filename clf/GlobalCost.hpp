#ifndef GLOBALCOST_HPP_
#define GLOBALCOST_HPP_

#include <boost/property_tree/ptree.hpp>

#include "clf/SupportPointCloud.hpp"
#include "clf/CostFunction.hpp"
#include "clf/DOFIndices.hpp"

namespace clf {

/// Compute the global cost of the coefficients for <em>all</em> of the support points in a clf::SupportPointCloud
class GlobalCost : public SparseCostFunction {
public:

  /**
  @param[in] cloud The support point cloud that holds all of the points (and their local cost functions)
  @param[in] pt The options for the global cost function
  */
  GlobalCost(std::shared_ptr<SupportPointCloud> const& cloud, boost::property_tree::ptree const& pt);

  virtual ~GlobalCost() = default;

  /// Get the uncoupled cost associated with the \f$i^{th}\f$ support point
  /**
  If \f$i\f$ is greater than the number of support points return a nullptr.
  @param[in] i We want the uncoupled cost associated with this support point
  \return The uncoupled cost associated with the \f$i^{th}\f$ support point
  */
  std::shared_ptr<UncoupledCost> GetUncoupledCost(std::size_t const i) const;

  /// Get the coupled costs associated with the \f$i^{th}\f$ support point
  /**
  If \f$i\f$ is greater than the number of support points return an empty vector.
  @param[in] i We want the coupled costs associated with this support point
  \return The vector of coupled costs associated with the \f$i^{th}\f$ support point
  */
  std::vector<std::shared_ptr<CoupledCost> > GetCoupledCost(std::size_t const i) const;

  /// Is this a quadratic cost function?
  /**
  The global cost is quadratic if all of the clf::UncoupledCost functions are quadratic
  \return <tt>true</tt>: The cost function is quadratic, <tt>false</tt>: The cost function is not quadratic
  */
  virtual bool IsQuadratic() const override;

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
  @param[in] beta The coefficients for all of the support points
  \return The \f$i^{th}\f$ entry is the \f$i^{th}\f$ sub-cost function \f$f_i(\boldsymbol{\beta})\f$.
  */
  Eigen::VectorXd CostImpl(Eigen::VectorXd const& beta) const;

  /// Compute the Jacobian matrix
  /**
  @param[in] beta The coefficients for all of the support points
  @param[out] jac The Jacobian matrix
  */
  void JacobianImpl(Eigen::VectorXd const& beta, Eigen::SparseMatrix<double>& jac) const;

private:

  /// Compute the number of sub-cost function
  /**
  The cost function takes the form:
  \f{equation*}{
  C = \min_{\boldsymbol{\beta} \in \mathbb{R}^{n}} \sum_{i=1}^{m} f_i(\boldsymbol{\beta})^{2}.
  \f}
  This function computes \f$m\f$ based on the uncoupled and coupled costs that are associated with each support point.
  @param[in] cloud The support point cloud that holds all of the points (and their local cost functions)
  \return The number of sub-cost functions
  */
  static std::size_t NumCostFunctions(std::shared_ptr<SupportPointCloud> const& cloud);

  /// The map from support point indices to the global degree of freedom indices
  const DOFIndices dofIndices;
};

} // namespace clf

#endif
