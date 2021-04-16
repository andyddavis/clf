#include "clf/SupportPointCloud.hpp"

namespace pt = boost::property_tree;
using namespace clf;

SupportPointCloud::SupportPointCloud(std::vector<std::shared_ptr<SupportPoint> > const& supportPoints, pt::ptree const& pt) : supportPoints(supportPoints) {
  // make sure the support points all of the same in/output dimensions
  CheckSupportPoints();

  // build the kd tree
  BuildKDTree(pt.get<std::size_t>("MaxLeaf", 10));
}

void SupportPointCloud::CheckSupportPoints() const {
  for( std::size_t i=1; i<supportPoints.size(); ++i ) {
    // check the input dimension
    if( supportPoints[i-1]->InputDimension()!=supportPoints[i]->InputDimension() ) { throw exceptions::SupportPointCloudDimensionException(exceptions::SupportPointCloudDimensionException::Type::INPUT, i-1, i); }

    // check the output dimension
    if( supportPoints[i-1]->OutputDimension()!=supportPoints[i]->OutputDimension() ) { throw exceptions::SupportPointCloudDimensionException(exceptions::SupportPointCloudDimensionException::Type::OUTPUT, i-1, i); }
  }
}

void SupportPointCloud::BuildKDTree(std::size_t const maxLeaf) {
  kdtree = std::make_shared<NanoflannKDTree>(InputDimension(), *this, nanoflann::KDTreeSingleIndexAdaptorParams(maxLeaf));
  kdtree->buildIndex();
}

std::shared_ptr<SupportPoint> SupportPointCloud::GetSupportPoint(std::size_t const i) const { return supportPoints[i]; }

std::size_t SupportPointCloud::NumSupportPoints() const { return supportPoints.size(); }

std::size_t SupportPointCloud::kdtree_get_point_count() const { return NumSupportPoints(); }

double SupportPointCloud::kdtree_get_pt(std::size_t const p, std::size_t const i) const {
  assert(p<supportPoints.size());
  return supportPoints[p]->x(i);
}

std::size_t SupportPointCloud::InputDimension() const {
  assert(supportPoints.size()>0);
  return supportPoints[0]->InputDimension();
}

std::size_t SupportPointCloud::OutputDimension() const {
  assert(supportPoints.size()>0);
  return supportPoints[0]->OutputDimension();
}
