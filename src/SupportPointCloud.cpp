#include "clf/SupportPointCloud.hpp"

namespace pt = boost::property_tree;
using namespace clf;

SupportPointCloud::SupportPointCloud(std::vector<std::shared_ptr<SupportPoint> > const& supportPoints, pt::ptree const& pt) : supportPoints(supportPoints), requireConnectedGraphs(pt.get<bool>("RequireConnectedGraphs", false)) {
  // make sure the support points all of the same in/output dimensions
  CheckSupportPoints();

  // build the kd tree
  BuildKDTree(pt.get<std::size_t>("MaxLeaf", 10));
}

std::shared_ptr<SupportPointCloud> SupportPointCloud::Construct(std::vector<std::shared_ptr<SupportPoint> > const& supportPoints, boost::property_tree::ptree const& pt) {
  // make the cloud
  auto cloud = std::shared_ptr<SupportPointCloud>(new SupportPointCloud(supportPoints, pt));

  // find the required nearest neighbors for each support point
  cloud->FindNearestNeighbors();

  return cloud;
}

void SupportPointCloud::FindNearestNeighbors() const {
  // loop through each support point
  for( const auto& point : supportPoints ) {
    assert(point);
    // the maximum required nearest neighbors
    const std::size_t numNeigh = point->NumNeighbors();
    if( numNeigh>supportPoints.size() ) { throw exceptions::SupportPointCloudNotEnoughPointsException(supportPoints.size(), numNeigh); }

    // find the nearest nieghbors
    std::vector<std::size_t> neighInd; std::vector<double> neighDist;
    FindNearestNeighbors(point->x, numNeigh, neighInd, neighDist);
    assert(neighInd.size()==numNeigh); assert(neighDist.size()==numNeigh);

    // set the nearest neighbors
    point->SetNearestNeighbors(shared_from_this(), neighInd, neighDist);
  }

  if( requireConnectedGraphs ) {
    // check to make sure the graph is connected
    if( !CheckConnected() ) { throw exceptions::SupportPointCloudNotConnected(); }
  }
}

bool SupportPointCloud::CheckConnected() const {
  std::vector<bool> visited(supportPoints.size(), false);

  CheckConnected(0, visited);

  for( const auto& node : visited ) { if( !node ) { return false; } }
  return true;
}

void SupportPointCloud::CheckConnected(std::size_t const ind, std::vector<bool>& visited) const {
  // mark the current node as visited
  assert(ind<visited.size());
  visited[ind] = true;

  // get the connected neighbors and recursively mark them as visited
  const std::vector<size_t> neighbors = supportPoints[ind]->GlobalNeighborIndices();
  for( std::size_t i=0; i<neighbors.size(); ++i ) {
    assert(neighbors[i]<visited.size());
    if( !visited[neighbors[i]] ) { CheckConnected(neighbors[i], visited); }
  }
}

void SupportPointCloud::CheckSupportPoints() const {
  for( std::size_t i=1; i<supportPoints.size(); ++i ) {
    // check the input dimension
    if( supportPoints[i-1]->model->inputDimension!=supportPoints[i]->model->inputDimension ) { throw exceptions::SupportPointCloudDimensionException(exceptions::SupportPointCloudDimensionException::Type::INPUT, i-1, i); }

    // check the output dimension
    if( supportPoints[i-1]->model->outputDimension!=supportPoints[i]->model->outputDimension ) { throw exceptions::SupportPointCloudDimensionException(exceptions::SupportPointCloudDimensionException::Type::OUTPUT, i-1, i); }
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
  return supportPoints[0]->model->inputDimension;
}

std::size_t SupportPointCloud::OutputDimension() const {
  assert(supportPoints.size()>0);
  return supportPoints[0]->model->outputDimension;
}

void SupportPointCloud::FindNearestNeighbors(Eigen::VectorXd const& point, std::size_t const k, std::vector<std::size_t>& neighInd, std::vector<double>& neighDist) const {
  assert(k<=supportPoints.size());

  // resize the input vecotrs
  assert(k>0);
  neighInd.resize(k); neighDist.resize(k);

  // find the nearest neighbors
  assert(kdtree);
  const std::size_t nfound = kdtree->knnSearch(point.data(), k, neighInd.data(), neighDist.data());
  assert(nfound==k);
}

std::pair<std::vector<std::size_t>, std::vector<double> > SupportPointCloud::FindNearestNeighbors(Eigen::VectorXd const& point, std::size_t const k) const {
    std::pair<std::vector<std::size_t>, std::vector<double> > result;
    SupportPointCloud::FindNearestNeighbors(point, k, result.first, result.second);
    return result;
  }

std::vector<std::shared_ptr<SupportPoint> >::const_iterator SupportPointCloud::Begin() const { return supportPoints.begin(); }

std::vector<std::shared_ptr<SupportPoint> >::const_iterator SupportPointCloud::End() const { return supportPoints.end(); }
