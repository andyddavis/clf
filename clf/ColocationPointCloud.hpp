#ifndef COLOCATIONPOINTCLOUD_HPP_
#define COLOCATIONPOINTCLOUD_HPP_

#include "clf/ColocationPoint.hpp"
#include "clf/SupportPointCloud.hpp"
#include "clf/ColocationPointSampler.hpp"

namespace clf {

/// Forward declaration of the colocation cost
class ColocationCost;

/*
<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"NumColocationPoints"   | <tt>std::size_t</tt> | <tt>The number of support points</tt> | The number of colocation points |
*/
class ColocationPointCloud {
public:

  /// Let the colocation cost be a friend
  friend ColocationCost;

  /**
  @param[in] sampler The distribution that allows us to sample colocation points
  @param[in] supportCloud The support point cloud associated with the colocation point cloud
  @param[in] pt Options for the collocation point cloud
  */
  ColocationPointCloud(std::shared_ptr<ColocationPointSampler> const& sampler, std::shared_ptr<SupportPointCloud> const& supportCloud, boost::property_tree::ptree const& pt);

  virtual ~ColocationPointCloud() = default;

  /// Resample the colocation points
  void Resample();

  /// Get the \f$i^{th}\f$ sample
  /**
  @param[in] i We want this sample
  \return The \f$i^{th}\f$ sample
  */
  std::shared_ptr<ColocationPoint> GetColocationPoint(std::size_t const i) const;

  /// An iterator to the first colocation point
  std::vector<std::shared_ptr<ColocationPoint> >::const_iterator Begin() const;

  /// An iterator to the last colocation point
  std::vector<std::shared_ptr<ColocationPoint> >::const_iterator End() const;

  /// The input dimension
  std::size_t InputDimension() const;

  /// The output dimension
  std::size_t OutputDimension() const;

  /// The number of colocation points
  const std::size_t numColocationPoints;

private:

  /// The sampler to generate new colocation points
  std::shared_ptr<ColocationPointSampler> sampler;

  /// The collocation points
  std::vector<std::shared_ptr<ColocationPoint> > points;

  /// The support point cloud that stores the local functions
  std::shared_ptr<SupportPointCloud> supportCloud;
};

} // namespace clf

#endif
