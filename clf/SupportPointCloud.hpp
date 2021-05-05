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
"RequireConnectedGraphs"   | <tt>bool</tt> | <tt>false</tt> | <tt>true</tt>: The graphs associated with each output must be connected, <tt>false</tt>: The graphs associated with each output need not be connected |
*/
class SupportPointCloud : public std::enable_shared_from_this<SupportPointCloud> {
// make the constructors protected because we will always need to find the nearest neighbors after construction
protected:

  /**
  @param[in] supportPoints The \f$i^{th}\f$ entry is the support point associated with \f$x^{(i)}\f$
  @param[in] pt The options for the support point cloud
  */
  SupportPointCloud(std::vector<std::shared_ptr<SupportPoint> > const& supportPoints, boost::property_tree::ptree const& pt);

public:

  virtual ~SupportPointCloud() = default;

  /// Construct the cloud and also set the nearest neighbors
  /**
  @param[in] supportPoints The \f$i^{th}\f$ entry is the support point associated with \f$x^{(i)}\f$
  @param[in] pt The options for the support point cloud
  */
  static std::shared_ptr<SupportPointCloud> Construct(std::vector<std::shared_ptr<SupportPoint> > const& supportPoints, boost::property_tree::ptree const& pt);

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

  /// Find the \f$k\f$ nearest neighbors
  /**
  @param[in] point We want to find the nearest neighbors of this point
  @param[in] k We want to find this many nearest neighbors
  \return First: The indices of the nearest neighbors, Second: The squared distances from the input point to its nearest neighbors
  */
  std::pair<std::vector<std::size_t>, std::vector<double> > FindNearestNeighbors(Eigen::VectorXd const& point, std::size_t const k) const;

  /// An iterator to the first support point
  std::vector<std::shared_ptr<SupportPoint> >::const_iterator Begin() const;

  /// An iterator to the last support point
  std::vector<std::shared_ptr<SupportPoint> >::const_iterator End() const;

  /// The total number of coefficients
  /**
  The sum of all of the number coefficients required by each support point.
  */
  const std::size_t numCoefficients;

private:

  /// Find the required nearest neighbors for each support point
  void FindNearestNeighbors() const;

  /// The total number of basis function coefficients
  /**
  \return The sum of the number of basis coefficients over all of the support point
  */
  static std::size_t NumCoefficients(std::vector<std::shared_ptr<SupportPoint> > const& supportPoints);

  /// Make sure the support points all have the same input/output dimension
  void CheckSupportPoints() const;

  /// Build the \f$k\f$-\f$d\f$ tree using the current support points
  /**
  @param[in] maxLeaf The max leaf parameter for the \f$k\f$-\f$d\f$ tree
  */
  void BuildKDTree(std::size_t const maxLeaf);

  /// Check to make sure the graph is connected
  /**
  \return <tt>true</tt>: The graph is connected, <tt>false</tt>: The graph is not connected
  */
  bool CheckConnected() const;

  /// Check to make sure the graph is connected
  /**
  @param[in] ind The current index as we transverse the graph
  @param[in] visited A vector of points on the graph, <tt>true</tt> if we have visited that node already, <tt>false</tt> if not
  */
  void CheckConnected(std::size_t const ind, std::vector<bool>& visited) const;

  /// The \f$i^{th}\f$ entry is the support point associated with \f$x^{(i)}\f$
  const std::vector<std::shared_ptr<SupportPoint> > supportPoints;

  /// The \f$k\f$-\f$d\f$ tree type that sorts the support points
  typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, SupportPointCloud>, SupportPointCloud> NanoflannKDTree;

  /// The \f$k\f$-\f$d\f$ tree used to compute nearest neighbors to any point \f$x\f$
  std::shared_ptr<NanoflannKDTree> kdtree;

  /// Require that the graph associated with each output is connected
  const bool requireConnectedGraphs;
};

} // namespace clf

#endif
