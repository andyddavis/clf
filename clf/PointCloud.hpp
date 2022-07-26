#ifndef POINTCLOUD_HPP_
#define POINTCLOUD_HPP_

#include "clf/Point.hpp"

namespace clf {

/// A cloud of points \f$ \{ x_i \}_{i=1}^{n} \f$
class PointCloud {
public:

  /// Create an empty point cloud
  PointCloud();

  virtual ~PointCloud() = default;

  /// The number of points in the point cloud \f$n\f$ 
  std::size_t NumPoints() const;

  /// Add a new point to the point cloud 
  /**
     @param[in] point We want to add this point to the point cloud
   */
  void AddPoint(Point const& point);

  /// Add a new point to the point cloud by constructing it 
  /**
     @param[in] point We want to add this point to the point cloud
   */
  void AddPoint(Eigen::VectorXd const& point);

  /// Get the \f$i^{\text{th}}\f$ point
  /**
     @param[in] ind The index of the point we want 
     \return The \f$i^{\text{th}}\f$ point
   */
  Point Get(std::size_t const ind) const;

private:

  /// A vector of points 
  std::vector<Point> points;
};

} // namespace clf

#endif
