#ifndef COLLOCATIONPOINTCLOUD_HPP_
#define COLLOCATIONPOINTCLOUD_HPP_

#include "clf/CollocationPoint.hpp"
#include "clf/SupportPointCloud.hpp"
#include "clf/CollocationPointSampler.hpp"

namespace clf {

/// Forward declaration of the collocation cost
class CollocationCost;

/// A collection of collocation points (see clf::CollocationPoint)
/*
<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"NumCollocationPoints"   | <tt>std::size_t</tt> | <tt>The number of support points</tt> | The number of colocation points |
*/
class CollocationPointCloud : public PointCloud {
public:

  /// Let the colocation cost be a friend
  friend CollocationCost;

  /**
  @param[in] sampler The distribution that allows us to sample collocation points
  @param[in] supportCloud The support point cloud associated with the collocation point cloud
  @param[in] pt Options for the collocation point cloud
  */
  CollocationPointCloud(std::shared_ptr<CollocationPointSampler> const& sampler, std::shared_ptr<SupportPointCloud> const& supportCloud, boost::property_tree::ptree const& pt);

  virtual ~CollocationPointCloud() = default;

  /// Resample the collocation points
  void Resample();

  /// Get the \f$i^{th}\f$ sample
  /**
  @param[in] i We want this sample
  \return The \f$i^{th}\f$ sample
  */
  std::shared_ptr<CollocationPoint> GetCollocationPoint(std::size_t const i) const;

  /// The number of colocation points
  const std::size_t numCollocationPoints;

private:

  /// The sampler to generate new collocation points
  std::shared_ptr<CollocationPointSampler> sampler;

  /// The support point cloud that stores the local functions
  std::shared_ptr<SupportPointCloud> supportCloud;
};

} // namespace clf

#endif
