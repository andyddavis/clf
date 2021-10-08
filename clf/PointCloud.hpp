#ifndef POINTCLOUD_HPP_
#define POINTCLOUD_HPP_

#include "clf/Point.hpp"

namespace clf {

/// A set of points \f$\{x_i\}_{i=1}^{n}\f$
class PointCloud {
public:

  /// Create an empty point cloud
  PointCloud();

  /// Create a point cloud given a set of points
  /**
  @param[in] points The points in the point cloud
  */
  PointCloud(std::vector<std::shared_ptr<Point> > const& points);

  virtual ~PointCloud() = default;

  /// The number of points
  /**
  \return The number of points
  */
  std::size_t NumPoints() const;

  /// An iterator to the first point
  std::vector<std::shared_ptr<Point> >::const_iterator Begin() const;

  /// An iterator to the last point
  std::vector<std::shared_ptr<Point> >::const_iterator End() const;

protected:

  /// Each entry is a point in the cloud
  std::vector<std::shared_ptr<Point> > points;

private:
};

} // namespace clf

#endif
