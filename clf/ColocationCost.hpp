#ifndef COLOCATIONCOST_HPP_
#define COLOCATIONCOST_HPP_

#include "clf/ColocationPointCloud.hpp"
#include "clf/CostFunction.hpp"

namespace clf {

/// The cost function to minimize the expected least squares error with respect to a given measure
/**
*/
class ColocationCost : public SparseCostFunction {
public:

  /**
  @param[in] colocationCloud The collocation cloud that we will use to comptue this cost
  */
  ColocationCost(std::shared_ptr<ColocationPointCloud> const& colocationCloud);

  virtual ~ColocationCost() = default;

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
  virtual double PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const override;

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
  std::shared_ptr<ColocationPointCloud> colocationCloud;
};

} // namespace clf

#endif
