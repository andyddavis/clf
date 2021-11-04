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

  /// Get the \f$i^{th}\f$ point
  /**
  @param[in] ind The index \f$i\f$ of the point we want
  \return The \f$i^{th}\f$ point
  */
  std::shared_ptr<Point> GetPoint(std::size_t const ind) const;

  /// An iterator to the first point
  std::vector<std::shared_ptr<Point> >::const_iterator Begin() const;

  /// An iterator to the last point
  std::vector<std::shared_ptr<Point> >::const_iterator End() const;

  /// Write the points to file
  /**
  @param filename The output file name (must end in <tt>".h5"</tt>)
  @param dataset The dataset in the hdf5 file where we store the points (defaults to root "/")
  */
  virtual void WriteToFile(std::string const& filename, std::string const& dataset = "/") const = 0;

protected:

  /// Each entry is a point in the cloud
  std::vector<std::shared_ptr<Point> > points;

private:
};

} // namespace clf

#endif
