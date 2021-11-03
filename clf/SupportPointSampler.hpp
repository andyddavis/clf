#ifndef SUPPORTPOINTSAMPLER_HPP_
#define SUPPORTPOINTSAMPLER_HPP_

#include "clf/PointSampler.hpp"
#include "clf/SupportPoint.hpp"

namespace clf {

/// Sample a support point (see clf::SupportPoint) from the distribution \f$\pi\f$
/**
<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"SupportPoint"   | <tt>boost::property_tree::ptree</tt> | --- | The options to create a clf::SupportPoint. |
*/
class SupportPointSampler : public PointSampler {
public:

  /**
  @param[in] randVar This allows us to sample from the distribution \f$\pi\f$
  @param[in] model The clf::Model associated with each support point
  @param[in] options Options for a clf::SupportPoint
  */
  SupportPointSampler(std::shared_ptr<muq::Modeling::RandomVariable> const& randVar, std::shared_ptr<Model> const& model, boost::property_tree::ptree const& options);

  /// Generate support points that use the identity model 
  /**
  <B>Configuration Parameters:</B>
  Parameter Key | Type | Default Value | Description |
  ------------- | ------------- | ------------- | ------------- |
  "OutputDimension"   | <tt>std::size_t</tt> | <tt>1</tt> | The output dimension \f$m\f$. |

  @param[in] randVar This allows us to sample from the distribution \f$\pi\f$
  @param[in] options Options for a clf::SupportPoint
  */
  SupportPointSampler(std::shared_ptr<muq::Modeling::RandomVariable> const& randVar, boost::property_tree::ptree const& options);

  virtual ~SupportPointSampler() = default;

  /// Sample a support point from the distribution \f$\pi\f$
  std::shared_ptr<SupportPoint> Sample() const;

private:

  /// Options used to construct each support point
  const boost::property_tree::ptree options;
};

} // namespace clf

#endif
