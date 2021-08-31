#ifndef COUPLEDSUPPORTPOINT_HPP_
#define COUPLEDSUPPORTPOINT_HPP_

#include "clf/SupportPoint.hpp"

namespace clf {

/// A clf::CoupledSupportPoint is a child of clf::SupportPoint, but its cost function couples it with its nearest neighbors.
/**
Let \f$\boldsymbol{x}\f$ be the location of this support point. The coupling function is
\f{equation*}{
f(\boldsymbol{x}, \boldsymbol{y}) = \begin{cases}
c_0 \exp{\left( c_1 \left( 1- \frac{1}{1-\|\boldsymbol{x} - \boldsymbol{y} \|^2/r_{\max}^2} \right) \right)} & \mbox{if } \|\boldsymbol{x} - \boldsymbol{y} \|^2<1 \\
0 & \mbox{else},
\end{cases}
\f}
where \f$r_{\max}\f$ is the distance from the point \f$\boldsymbol{x}\f$ to its furtherest neighbor.

<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"MagnitudeScale"   | <tt>double</tt> | <tt>1.0</tt> | The parameter \f$c_0\f$ |
"ExponentialScaling"   | <tt>double</tt> | <tt>1.0</tt> | The parameter \f$c_1\f$ |
*/
class CoupledSupportPoint : public SupportPoint {
protected:

  /// SupportPoint must be a friend so that its static Construct method can call this constructor
  friend SupportPoint;

  /**
  @param[in] x The location of the support point \f$x\f$
  @param[in] model The model that defines the "data" at this support point
  @param[in] pt The options for the support point
  */
  CoupledSupportPoint(Eigen::VectorXd const& x, std::shared_ptr<const Model> const& model, boost::property_tree::ptree const& pt);

public:

  virtual ~CoupledSupportPoint() = default;

  /// A static construct method
  /**
  @param[in] x The location of the support point \f$x\f$
  @param[in] model The model that defines the "data" at this support point
  @param[in] pt The options for the support point
  \return A smart pointer to the support point
  */
  static std::shared_ptr<CoupledSupportPoint> Construct(Eigen::VectorXd const& x, std::shared_ptr<const Model> const& model, boost::property_tree::ptree const& pt);

  /// Determines the coupling coefficient between this support point and its neighbor
  /**
  The coupling function is
  \f{equation*}{
  f(\boldsymbol{x}, \boldsymbol{y}) = \begin{cases}
  c_0 \exp{\left( c_1 \left( 1-\frac{1}{1-\|\boldsymbol{x} - \boldsymbol{y} \|^2/r_{\max}^2}\right)\right)} & \mbox{if } \|\boldsymbol{x} - \boldsymbol{y} \|^2<1 \\
  0 & \mbox{else},
  \end{cases}
  \f}
  where \f$r_{\max}\f$ is the distance from the point \f$\boldsymbol{x}\f$ to its furtherest neighbor.

  @param[in] neighInd The local index of the nearest neighbor
  \return The coupling coefficient between this support point and its neighbor
  */
  virtual double CouplingFunction(std::size_t const neighInd) const override;

private:

  /// The parameter \f$c_0\f$ in clf::CoupledSupportPoint::CouplingFunction
  /**
  The default value is \f$1\f$
  */
  const double c0;

  /// The parameter \f$c_1\f$ in clf::CoupledSupportPoint::CouplingFunction
  /**
  The default value is \f$1\f$
  */
  const double c1;
};

} // namespace clf

#endif
