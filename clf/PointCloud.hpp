#ifndef POINTCLOUD_HPP_
#define POINTCLOUD_HPP_

#include <memory>

#include "clf/Point.hpp"

#include "clf/Domain.hpp"

namespace clf {

/// A cloud of points \f$ \{ x_i \}_{i=1}^{n} \f$
class PointCloud {
public:

  /// Create an empty point cloud
  /**
     @param[in] The domain containing the points (defaults to the null pointer)
   */
  PointCloud(std::shared_ptr<Domain> const& domain = nullptr);

  virtual ~PointCloud() = default;

  /// The number of points in the point cloud \f$n\f$ 
  std::size_t NumPoints() const;

  /// Add a new point to the point cloud 
  /**
     @param[in] point We want to add this point to the point cloud
   */
  void AddPoint(std::shared_ptr<Point> const& point);

  /// Add a new point to the point cloud by sampling the domain
  void AddPoint();

  /// Add multiple new points to the point cloud by sampling the domain
  /**
     @parma[in] n The number of points to add (defaults to 1)
   */
  void AddPoints(std::size_t const n = 1);

  /// Get the \f$i^{\text{th}}\f$ point
  /**
     @param[in] ind The index of the point we want 
     \return The \f$i^{\text{th}}\f$ point
   */
  std::shared_ptr<Point> Get(std::size_t const ind) const;

  /// Get the point with a given global id (see clf::Point::id)
  /**
     The point must be in this cloud
     @param[in] id The id of the desired point
     \return The point with the given global id
   */
  std::shared_ptr<Point> GetUsingID(std::size_t const id) const;

  /// Get the distance between two points in the cloud
  /**
     @param[in] ind1 The index of the first point
     @param[in] ind2 The index of the second point
     \return The distance between the points 
   */
  double Distance(std::size_t const ind1, std::size_t const ind2) const;

  /// Get the index of the closest point in cloud to a given point
  /**
     @param[in] x The point 
     \return First: The index of closest point in the cloud to \f$x\f$, Second: The distance to the point
   */
  std::pair<std::size_t, double> ClosestPoint(Eigen::VectorXd const& x) const;

  /// Find the \f$k\f$ nearest neighbors of the \f$i^{\text{th}}\f$ point
  /**
     @param[in] ind The index of the point whose nearest neighbors we are looking for
     @param[in] k We want the \f$k\f$ nearest neighbors 
     \return A vector of the \f$k\f$ nearest neighbors
   */
  std::vector<std::size_t> NearestNeighbors(std::size_t const ind, std::size_t const k) const;

private:

  std::size_t IndexFromID(std::size_t const id) const;

  void UpdateNeighbors(std::size_t const id, std::size_t const neigh, double const dist) const;

  /// The domain containing the points
  /**
     This pointer may be null, which indicates that we just have a bunch of points but no context for where they live. For example, we wouldn't know a norm or distance metric.
   */
  std::shared_ptr<Domain> domain;

  /// A vector of points 
  std::vector<std::shared_ptr<Point> > points;

  /// The <tt>std::unordered_map</tt> needs a hash to have a std::pair<std::size_t, std::size_t></tt> as a key
  struct PairHash {
    /// The hash for std::pair<std::size_t, std::size_t></tt>
    std::size_t operator()(std::pair<std::size_t, std::size_t> const& p) const;
  };

  /// The type for a map from a pair of points to the distance between them
  typedef std::unordered_map<std::pair<std::size_t, std::size_t>, double, PairHash> DistanceMap;

  /// A map from a pair of points to the distance between them
  mutable DistanceMap distances;

  /// The type for a map from a point to its nearest neighbors (and the distance to each point)
  /**
     The vector of nearest neighbors is sorted by distance
   */
  typedef std::unordered_map<std::size_t, std::vector<std::size_t> > NeighborMap;

  /// A map from a point to its nearest neighbors (and the distance to each point)
  mutable NeighborMap neighbors;
};

} // namespace clf

#endif
