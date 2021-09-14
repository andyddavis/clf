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

private:

  /// Implement the sub-cost functions
  /**
  @param[in] data When mapped into a matrix, each column is the function we are tryng to approximate evaluated at the support point
  */
  virtual Eigen::VectorXd CostImpl(Eigen::VectorXd const& data) const override;

  /**
  @param[in] data When mapped into a matrix, each column is the function we are tryng to approximate evaluated at the support point
  */
  virtual void JacobianImpl(Eigen::VectorXd const& data, Eigen::SparseMatrix<double>& jac) const override;

  /// The collocation cloud that we will use to comptue this cost
  std::shared_ptr<ColocationPointCloud> colocationCloud;
};

} // namespace clf

#endif
