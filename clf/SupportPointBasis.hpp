#ifndef SUPPORTPOINTBASIS_HPP_
#define SUPPORTPOINTBASIS_HPP_

#include "clf/BasisFunctions.hpp"

namespace clf {

/// Forward declaration of a clf::SupportPoint
class SupportPoint;

/// A wrapper around a generic basis function that scales the evaluation into "local" coordinates around a support point
/**
Given basis functions \f$\{\hat{\phi}_i\}_{i=1}^{n}\f$ (e.g., clf::SinCosBasis or clf::PolynomialBasis) and a clf::SupportPoint located at \f$y\f$ with radius \f$\delta\f$ (see clf::SupportPoint::Radius) this class defines scaled basis functions \f$\phi_i(x) = \hat{\phi}_i((x-y)/\delta)\f$.

<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"Radius"   | <tt>double</tt> | <tt>1.0</tt> | The initial value of the \f$\delta\f$ parameter. |
*/
class SupportPointBasis : public BasisFunctions {
public:
  /**
  @param[in] point The support point associated with this basis
  @param[in] basis The unscaled basis functions
  @param[in] pt The options for the basis functions
  */
  SupportPointBasis(std::shared_ptr<SupportPoint> const& point, std::shared_ptr<const BasisFunctions> const& basis, boost::property_tree::ptree const& pt);

  virtual ~SupportPointBasis() = default;

  /// The radius of the ball the defines where the local function is relatively accurate
  /**
  \return The radius of the ball the defines where the local function is relatively accurate
  */
  double Radius() const;

  /// The radius of the ball the defines where the local function is relatively accurate
  /**
  \return The radius of the ball the defines where the local function is relatively accurate
  */
  void SetRadius(double const newdelta) const;

  /// Convert from local to global coordinates
  /**
  @param[in] x The global coordinate
  \return The local coordinate \f$\hat{x} = (x-y)/\delta\f$
  */
  Eigen::VectorXd LocalCoordinate(Eigen::VectorXd const& x) const;

  /// Convert from global to local coordinates
  /**
  @param[in] xhat The local coordinate
  \return The global coordinate \f$x = \delta \hat{x} + y\f$
  */
  Eigen::VectorXd GlobalCoordinate(Eigen::VectorXd const& xhat) const;

  /// The unscaled basis functions
  const std::shared_ptr<const BasisFunctions> basis;

protected:

  /// Evaluate the scalar basis function \f$l_i: \mathbb{R} \mapsto \mathbb{R}\f$.
  /**
  The basis function is defined by the product of these scalar functions. This must be implemented by a child.
  @param[in] x The point where we are evaluating the scalar basis function
  @param[in] ind The index of the \f$i^{th}\f$ scalar basis function
  @param[in] coordinate The index of the coordinate (the parameter \$x\f$ is the \f$i^{th}\f$ input)
  \return The scalar basis function evaluation
  */
  virtual double ScalarBasisFunction(double const x, std::size_t const ind, std::size_t coordinate) const override;

  /// Evaluate the \f$k^{th}\f$ derivative of the scalar basis function \f$\frac{d^k l_i}{d x^{k}}\f$.
  /**
  This must be implemented by a child.
  @param[in] x The point where we are evaluating the scalar basis function
  @param[in] ind The index of the \f$i^{th}\f$ scalar basis function
  @param[in] coordinate The index of the coordinate (the parameter \$x\f$ is the \f$i^{th}\f$ input)
  @param[in] k We want the \f$k^{th}\f$ derivative
  \return The scalar basis function evaluation
  */
  virtual double ScalarBasisFunctionDerivative(double const x, std::size_t const ind, std::size_t const coordinate, std::size_t const k) const override;

private:

  /// Transformation a single coordinate into the local coordinates
  /**
  @param[in] xi The \f$i^{th}\f$ global coordinate
  @param[in] coord The coordinate \f$i\f$
  \return xhati The \f$i^{th}\f$ local coordinate
  */
  double LocalCoordinate(double const xi, std::size_t const coord) const;

  /// The parameter \f$\delta\f$ that defines the radius for which we expect this local function to be relatively accurate
  /**
  The default value is \f$\delta=1\f$. This parameter defines the local coordinate transformation \f$\hat{x}(y) = (y-x)/\delta\f$.
  */
  mutable double delta;

  /// The support point associated with this basis
  std::weak_ptr<const SupportPoint> point;

};

} // namespace clf

#endif
