#include "clf/PointCloud.hpp"

using namespace clf;

PointCloud::PointCloud() {}

std::size_t PointCloud::NumPoints() const { return points.size(); }

void PointCloud::AddPoint(Point const& point) { 
  if( points.size()>0 ) { assert(point.x.size()==points[0].x.size()); }
  points.push_back(point); 
}

void PointCloud::AddPoint(Eigen::VectorXd const& point) {
  if( points.size()>0 ) { assert(point.size()==points[0].x.size()); }
  points.emplace_back(point); 
}

Point PointCloud::Get(std::size_t const ind) const { 
  assert(ind<points.size()); 
  return points[ind];
}
