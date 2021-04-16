#ifndef SUPPORTPOINTCLOUD_HPP_
#define SUPPORTPOINTCLOUD_HPP_

#include <nanoflann.hpp>

#include "clf/SupportPoint.hpp"
#include "clf/SupportPointCloudExceptions.hpp"

namespace clf {

/// The cloud of support points \f$\{x^{(i)}\}_{i=1}^{n}\f$ that defines the coupled local function
/**
<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"MaxLeaf"   | <tt>std::size_t</tt> | <tt>10</tt> | The max leaf parameter for constructing the \f$k\f$-\f$d\f$ tree. |
*/
class SupportPointCloud {
public:

  /**
  @param[in] supportPoints The \f$i^{th}\f$ entry is the support point associated with \f$x^{(i)}\f$
  @param[in] pt The options for the support point cloud
  */
  SupportPointCloud(std::vector<std::shared_ptr<SupportPoint> > const& supportPoints, boost::property_tree::ptree const& pt);

  virtual ~SupportPointCloud() = default;

  /// Get the number of support points
  /**
  Required by <tt>nanoflann</tt> to construct the \f$k\f$-\f$d\f$ tree
  \return The number of points
  */
  std::size_t kdtree_get_point_count() const;

  /// Get the \f$i^{th}\f$ component of the \f$p^{th}\f$ point
  /**
  Required by <tt>nanoflann</tt> to construct the \f$k\f$-\f$d\f$ tree
  \return The number of points
  @param[in] p We want to access this support point number
  @param[in] i We want this index of the support point location
  \return The \f$i^{th}\f$ component of the \f$p^{th}\f$ point
  */
  double kdtree_get_pt(std::size_t const p, std::size_t const i) const;

  /// Optional bounding-box computation
  /**
  Required by <tt>nanoflann</tt> to construct the \f$k\f$-\f$d\f$ tree
  \return The number of points
  \return Return <tt>false</tt> to default to a standard bounding box computation loop
  */
  template<class BBOX>
  inline bool kdtree_get_bbox(BBOX& bb) const { return false; }

  /// The number of support points
  /**
  \return The number of support points
  */
  std::size_t NumSupportPoints() const;

  /// Get the \f$i^{th}\f$ support point
  /**
  \return The \f$i^{th}\f$ support point
  */
  std::shared_ptr<SupportPoint> GetSupportPoint(std::size_t const i) const;

  /// The input dimension for each support point
  /**
  \return The input dimension for each support point
  */
  std::size_t InputDimension() const;

  /// The output dimension for each support point
  /**
  \return The output dimension for each support point
  */
  std::size_t OutputDimension() const;

  /// Find the \f$k\f$ nearest neighbors
  /**
  @param[in] point We want to find the nearest neighbors of this point
  @param[in] k We want to find this many nearest neighbors
  @param[out] neighInd The indices of the nearest neighbors
  @param[out] neighDist The squared distances from the input point to its nearest neighbors
  */
  void FindNearestNeighbors(Eigen::VectorXd const& point, std::size_t const k, std::vector<std::size_t>& neighInd, std::vector<double>& neighDist) const;

private:

  /// Make sure the support points all have the same input/output dimension
  void CheckSupportPoints() const;

  /// Build the \f$k\f$-\f$d\f$ tree using the current support points
  /**
  @param[in] maxLeaf The max leaf parameter for the \f$k\f$-\f$d\f$ tree
  */
  void BuildKDTree(std::size_t const maxLeaf);

  /// The \f$i^{th}\f$ entry is the support point associated with \f$x^{(i)}\f$
  const std::vector<std::shared_ptr<SupportPoint> > supportPoints;

  /// The \f$k\f$-\f$d\f$ tree type that sorts the support points
  typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, SupportPointCloud>, SupportPointCloud> NanoflannKDTree;

  /// The \f$k\f$-\f$d\f$ tree used to compute nearest neighbors to any point \f$x\f$
  std::shared_ptr<NanoflannKDTree> kdtree;
};

} // namespace clf

#endif
