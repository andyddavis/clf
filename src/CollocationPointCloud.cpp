#include "clf/CollocationPointCloud.hpp"

#include <MUQ/Utilities/HDF5/HDF5File.h>

namespace pt = boost::property_tree;
using namespace muq::Utilities;
using namespace clf;

CollocationPointCloud::CollocationPointCloud(std::shared_ptr<CollocationPointSampler> const& sampler, std::shared_ptr<SupportPointCloud> const& supportCloud, pt::ptree const& pt) :
PointCloud(),
sampler(sampler),
supportCloud(supportCloud)
{
  points.resize(pt.get<std::size_t>("NumCollocationPoints", supportCloud->NumPoints()));
  Resample();
}

void CollocationPointCloud::Resample() {
  // reset the number of collocation points per support point to zero
  collocationPerSupport.resize(supportCloud->NumPoints(), std::vector<std::size_t>());

  const std::size_t nc = NumPoints();
  for( std::size_t i=0; i<nc; ++i ) {
    points[i] = sampler->Sample(i, nc);
    auto point = std::dynamic_pointer_cast<CollocationPoint>(points[i]);
    assert(point);

    // find the nearest support point and increment the number of collocation points associated with that point
    point->supportPoint = supportCloud->NearestSupportPoint(point->x);
    point->localIndex = collocationPerSupport[point->supportPoint.lock()->GlobalIndex()].size();
    collocationPerSupport[point->supportPoint.lock()->GlobalIndex()].push_back(i);
  }
}

std::size_t CollocationPointCloud::GlobalIndex(std::size_t const local, std::size_t const global) const {
  assert(global<collocationPerSupport.size());
  assert(local<collocationPerSupport[global].size());
  return collocationPerSupport[global][local];
}

std::shared_ptr<CollocationPoint> CollocationPointCloud::GetCollocationPoint(std::size_t const i) const {
  auto point = std::dynamic_pointer_cast<CollocationPoint>(points[i]);
  assert(point);
  return point;
}

std::size_t CollocationPointCloud::NumCollocationPerSupport(std::size_t const ind) const {
  assert(ind<supportCloud->NumPoints());
  return collocationPerSupport[ind].size();
}

std::vector<std::shared_ptr<CollocationPoint> > CollocationPointCloud::CollocationPerSupport(std::size_t const ind) const {
  assert(ind<supportCloud->NumPoints());
  const std::size_t num = NumCollocationPerSupport(ind);
  std::vector<std::shared_ptr<CollocationPoint> > pnts(num);
  for( std::size_t i=0; i<num; ++i ) { pnts[i] = GetCollocationPoint(collocationPerSupport[ind][i]); }
  return pnts;
}

void CollocationPointCloud::WriteToFile(std::string const& filename, std::string const& dataset) const {
  if( points.size()==0 ) { return; }

  assert(filename.size()>3);
  assert(filename.substr(filename.size()-3)==".h5");
  assert(dataset.size()>=1);
  assert(dataset.substr(0)=="/");

  std::vector<Eigen::MatrixXd> pnts(supportCloud->NumPoints());
  for( std::size_t i=0; i<supportCloud->NumPoints(); ++i ) { pnts[i].resize(NumCollocationPerSupport(i), supportCloud->GetPoint(i)->x.size()); }

  for( auto it=Begin(); it!=End(); ++it ) {
    auto point = std::dynamic_pointer_cast<CollocationPoint>(*it);
    assert(point);

    auto support = point->supportPoint.lock();
    pnts[support->GlobalIndex()].row(point->localIndex) = point->x;
  }

  HDF5File file(filename);
  for( std::size_t i=0; i<supportCloud->NumPoints(); ++i ) { 
    if( pnts[i].size()>0 ) { file.WriteMatrix(dataset+"/collocation points/support point "+std::to_string(i), pnts[i]); }
  }
  file.Close();
}
