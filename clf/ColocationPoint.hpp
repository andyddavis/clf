#ifndef COLOCATIONPOINT_HPP_
#define COLOCATIONPOINT_HPP_

#include "clf/SupportPoint.hpp"

namespace clf {

class ColocationPoint : public Point {
public:

  /**
  @param[in] x The location of the support point \f$x\f$
  @param[in] model The model that defines the "data" at this support point
  */
  ColocationPoint(Eigen::VectorXd const& x, std::shared_ptr<const Model> const& model);

  virtual ~ColocationPoint() = default;

  /// Evaluate the operator applied to the local function at the colocation point location
  /**
  @param[in] loc The location where we are evaluating the action of the operator
  */
  virtual Eigen::VectorXd Operator() const override;

  /// Evaluate the operator applied to the local function at a given point
  /**
  @param[in] loc The location where we are evaluating the action of the operator
  */
  virtual Eigen::VectorXd Operator(Eigen::VectorXd const& loc) const override;

  /// Evaluate the operator applied to the local function at a given point
  /**
  @param[in] loc The location where we are evaluating the action of the operator
  @param[in] coeffs The coefficients that define the local function
  */
  virtual Eigen::VectorXd Operator(Eigen::VectorXd const& loc, Eigen::VectorXd const& coeffs) const override;

  /// Evaluate the Jacobian of the operator applied to the local function at the point's location with the stored coefficients
  /**
  \return The model Jacobian with respect to the coefficeints
  */
  virtual Eigen::MatrixXd OperatorJacobian() const override;

  /// Evaluate the Jacobian of the operator applied to the local function at a given point with the stored coefficients
  /**
  @param[in] loc The location where we are evaluating the action of the operator
  \return The model Jacobian with respect to the coefficeints
  */
  virtual Eigen::MatrixXd OperatorJacobian(Eigen::VectorXd const& loc) const override;

  /// Evaluate the Jacobian of the operator applied to the local function at a given point
  /**
  @param[in] loc The location where we are evaluating the action of the operator
  @param[in] coeffs The coefficients that define the local function
  \return The model Jacobian with respect to the coefficeints
  */
  Eigen::MatrixXd OperatorJacobian(Eigen::VectorXd const& loc, Eigen::VectorXd const& coeffs) const;

  /// The nearest support point to this collocation point
  std::weak_ptr<SupportPoint> supportPoint;

private:
};

} // namespace clf

#endif
