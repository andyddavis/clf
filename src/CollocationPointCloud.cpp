#include "clf/CollocationPointCloud.hpp"

namespace pt = boost::property_tree;
using namespace clf;

CollocationPointCloud::CollocationPointCloud(std::shared_ptr<CollocationPointSampler> const& sampler, std::shared_ptr<SupportPointCloud> const& supportCloud, pt::ptree const& pt) :
PointCloud(),
sampler(sampler),
supportCloud(supportCloud),
numCollocationPoints(pt.get<std::size_t>("NumCollocationPoints", supportCloud->NumPoints()))
{
  points.resize(numCollocationPoints);
}

void CollocationPointCloud::Resample() {
  assert(points.size()==numCollocationPoints);
  for( auto& it : points ) {
    it = sampler->Sample();
    auto point = std::dynamic_pointer_cast<CollocationPoint>(it);
    assert(point);

    point->supportPoint = supportCloud->NearestSupportPoint(it->x);
  }
}

std::shared_ptr<CollocationPoint> CollocationPointCloud::GetCollocationPoint(std::size_t const i) const {
  auto point = std::dynamic_pointer_cast<CollocationPoint>(points[i]);
  assert(point);
  return point;
}
