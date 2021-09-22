#ifndef POINTCLOUD_HPP_
#define POINTCLOUD_HPP_

#include "clf/Point.hpp"

namespace clf {

/// A set of points \f$\{x_i\}_{i=1}^{n}\f$
class PointCloud {
public:

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
  const std::vector<std::shared_ptr<Point> > points;

private:
};

} // namespace clf

#endif
