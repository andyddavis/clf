#ifndef COLLOCATIONPOINT_HPP_
#define COLLOCATIONPOINT_HPP_

#include "clf/SupportPoint.hpp"

namespace clf {

/// Forward declaration of the collocation point cloud
class CollocationPointCloud;

/// A collocation point \f$x_i\f$ evaluates the forward model at the nearest clf::SupportPoint
class CollocationPoint : public Point {
public:

  /// The collocation point cloud is a friend
  friend CollocationPointCloud;

  /**
  @param[in] weight The weight of this collocation point
  @param[in] x The location of the support point \f$x\f$
  @param[in] model The model that defines the "data" at this support point
  */
  CollocationPoint(double const weight, Eigen::VectorXd const& x, std::shared_ptr<const Model> const& model);

  virtual ~CollocationPoint() = default;

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

  /// Evaluate the Jacobian of the operator using finite differences applied to the local function at a given point
  /**
  @param[in] loc The location where we are evaluating the action of the operator
  @param[in] coeffs The coefficients that define the local function
  @param[in] order The order of the finite difference approximation
  @param[in] fdEps The step size for the finite difference approximation (defaults to <tt>1.0e-6</tt>)
  \return The model Jacobian with respect to the coefficeints
  */
  Eigen::MatrixXd OperatorJacobianByFD(Eigen::VectorXd const& loc, Eigen::VectorXd const& coeffs, Model::FDOrder const order = Model::FDOrder::FIRST_UPWARD, double const fdEps = 1.0e-6) const;

  /// Evaluate the Hessian of the operator applied to the local function at the point's location with the stored coefficients
  /**
  \return Each component is the Hessian of the \f$j^{th}\f$ ouput with respect to the coefficeints
  */
  virtual std::vector<Eigen::MatrixXd> OperatorHessian() const override;

  /// Evaluate the Hessian of the operator applied to the local function at a given point with the stored coefficients
  /**
  @param[in] loc The location where we are evaluating the action of the operator
  \return Each component is the Hessian of the \f$j^{th}\f$ ouput with respect to the coefficeints
  */
  virtual std::vector<Eigen::MatrixXd> OperatorHessian(Eigen::VectorXd const& loc) const override;

  /// Evaluate the Hessian of the operator applied to the local function at a given point
  /**
  @param[in] loc The location where we are evaluating the action of the operator
  @param[in] coeffs The coefficients that define the local function
  \return Each component is the Hessian of the \f$j^{th}\f$ ouput with respect to the coefficeints
  */
  virtual std::vector<Eigen::MatrixXd> OperatorHessian(Eigen::VectorXd const& loc, Eigen::VectorXd const& coeffs) const override;

  /// The local index of this collocation point
  /**
  \return The local index
  */
  std::size_t LocalIndex() const;

  /// The nearest support point to this collocation point
  std::weak_ptr<SupportPoint> supportPoint;

  /// The weight of this collocation point
  const double weight;

private:

  /// The local index of this collocation point
  /**
  Each collocation point is associated with a support point; this support point is the nearest support point. The local index \f$k\f$ indicates that this is the \f$k^{th}\f$ collocation point associated with that particular support point.

  The local indices are not ordered in any way.
  */
  std::size_t localIndex;
};

} // namespace clf

#endif
