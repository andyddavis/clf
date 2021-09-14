#ifndef POINT_HPP_
#define POINT_HPP_

#include <Eigen/Core>

#include "clf/Model.hpp"

namespace clf {

/// A generic point in the domain
class Point {
public:

  /**
  @param[in] x The location of the support point \f$x\f$
  */
  Point(Eigen::VectorXd const& x);

  /**
  @param[in] x The location of the support point \f$x\f$
  @param[in] model The model that defines the "data" at this support point
  */
  Point(Eigen::VectorXd const& x, std::shared_ptr<const Model> const& model);

  virtual ~Point() = default;

  /// Evaluate the right hand side at this point's location
  /**
  \return The right hand side at this point's location
  */
  Eigen::VectorXd RightHandSide() const;

  /// Evaluate the right hand side at a given location
  /**
  @param[in] loc The location where we want to evaluate the right hand side
  \return The right hand side at a given location
  */
  Eigen::VectorXd RightHandSide(Eigen::VectorXd const& loc) const;

  /// Evaluate the operator applied to the local function at the point's location
  virtual Eigen::VectorXd Operator() const = 0;

  /// Evaluate the operator applied to the local function at a given point
  /**
  @param[in] loc The location where we are evaluating the action of the operator
  */
  virtual Eigen::VectorXd Operator(Eigen::VectorXd const& loc) const = 0;

  /// Evaluate the operator applied to the local function at a given point
  /**
  @param[in] loc The location where we are evaluating the action of the operator
  @param[in] coeffs The coefficients that define the local function
  */
  virtual Eigen::VectorXd Operator(Eigen::VectorXd const& loc, Eigen::VectorXd const& coeffs) const = 0;

  /// Evaluate the Jacobian of the operator applied to the local function at the point's location with the stored coefficients
  /**
  \return The model Jacobian with respect to the coefficeints
  */
  virtual Eigen::MatrixXd OperatorJacobian() const = 0;

  /// Evaluate the Jacobian of the operator applied to the local function at a given point with the stored coefficients
  /**
  @param[in] loc The location where we are evaluating the action of the operator
  \return The model Jacobian with respect to the coefficeints
  */
  virtual Eigen::MatrixXd OperatorJacobian(Eigen::VectorXd const& loc) const = 0;

  /// Evaluate the Jacobian of the operator applied to the local function at a given point
  /**
  @param[in] loc The location where we are evaluating the action of the operator
  @param[in] coefficients The coefficients that define the local function
  \return The model Jacobian with respect to the coefficeints
  */
  virtual Eigen::MatrixXd OperatorJacobian(Eigen::VectorXd const& loc, Eigen::VectorXd const& coefficients) const = 0;

  /// The location of the support point \f$x\f$.
  const Eigen::VectorXd x;

  /// The model that defines the data/observations at this support point
  const std::shared_ptr<const Model> model;

private:
};

} // namespace clf

#endif
