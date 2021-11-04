#ifndef COLLOCATIONPOINTCLOUD_HPP_
#define COLLOCATIONPOINTCLOUD_HPP_

#include "clf/CollocationPoint.hpp"
#include "clf/SupportPointCloud.hpp"
#include "clf/CollocationPointSampler.hpp"

namespace clf {

/// A collection of collocation points (see clf::CollocationPoint)
/**
<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"NumCollocationPoints"   | <tt>std::size_t</tt> | The number of support points | The number of colocation points |
*/
class CollocationPointCloud : public PointCloud {
public:

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

  /// Write the points to file
  /**
  Write all of the points into a location called <tt>"collocation points"</tt>. Inside this location there are data sets called <tt>"collocation points/support point i"</tt> such that each row is a collocation point associated with the \f$i^{th}\f$ support point. Empty matrices indicate that there are no collocation points associated with that support point.
  @param filename The output file name (must end in <tt>".h5"</tt>)
  @param dataset The dataset in the hdf5 file where we store the points (defaults to root "/")
  */
  virtual void WriteToFile(std::string const& filename, std::string const& dataset = "/") const override;

  /// The number of collocation points associated with the \f$i^{th}\f$ support point
  /**
  @param[in] ind The global index of the support point
  \return The number of collocation points associated with that support point
  */
  std::size_t NumCollocationPerSupport(std::size_t const ind) const;

  /// The global index of the collocation point given the local index and global index of the associated support point
  /**
  @param[in] local The local index of the collocation point
  @param[in] global The global index of the associated support point
  \return The global index
  */
  std::size_t GlobalIndex(std::size_t const local, std::size_t const global) const;

private:

  /// The sampler to generate new collocation points
  std::shared_ptr<CollocationPointSampler> sampler;

  /// The support point cloud that stores the local functions
  std::shared_ptr<SupportPointCloud> supportCloud;

  /// The global index of the collocation points such that the \f$i^{th}\f$ support point is its nearest neighbor
  std::vector<std::vector<std::size_t> > collocationPerSupport;
};

} // namespace clf

#endif
