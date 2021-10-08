#include "clf/PointCloud.hpp"

using namespace clf;

PointCloud::PointCloud() {}

PointCloud::PointCloud(std::vector<std::shared_ptr<Point> > const& points) :
points(points)
{}

std::vector<std::shared_ptr<Point> >::const_iterator PointCloud::Begin() const { return points.begin(); }

std::vector<std::shared_ptr<Point> >::const_iterator PointCloud::End() const { return points.end(); }

std::size_t PointCloud::NumPoints() const { return points.size(); }
