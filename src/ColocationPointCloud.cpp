#include "clf/ColocationPointCloud.hpp"

namespace pt = boost::property_tree;
using namespace clf;

ColocationPointCloud::ColocationPointCloud(std::shared_ptr<ColocationPointSampler> const& sampler, std::shared_ptr<SupportPointCloud> const& supportCloud, pt::ptree const& pt) :
sampler(sampler),
supportCloud(supportCloud),
numColocationPoints(pt.get<std::size_t>("NumColocationPoints", supportCloud->NumPoints()))
{
  points.resize(numColocationPoints);
}

void ColocationPointCloud::Resample() {
  assert(points.size()==numColocationPoints);
  for( auto& it : points ) {
    it = sampler->Sample();
    it->supportPoint = supportCloud->NearestSupportPoint(it->x);
  }
}

std::shared_ptr<ColocationPoint> ColocationPointCloud::GetColocationPoint(std::size_t const i) const {
  assert(i<points.size());
  return points[i];
}

std::vector<std::shared_ptr<ColocationPoint> >::const_iterator ColocationPointCloud::Begin() const { return points.begin(); }

std::vector<std::shared_ptr<ColocationPoint> >::const_iterator ColocationPointCloud::End() const { return points.end(); }

std::size_t ColocationPointCloud::InputDimension() const { return sampler->InputDimension(); }

std::size_t ColocationPointCloud::OutputDimension() const { return sampler->OutputDimension(); }
