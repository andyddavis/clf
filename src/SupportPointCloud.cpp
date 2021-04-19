#include "clf/SupportPointCloud.hpp"

namespace pt = boost::property_tree;
using namespace clf;

SupportPointCloud::SupportPointCloud(std::vector<std::shared_ptr<SupportPoint> > const& supportPoints, pt::ptree const& pt) : supportPoints(supportPoints), requireConnectedGraphs(pt.get<bool>("RequireConnectedGraphs", false)) {
  // make sure the support points all of the same in/output dimensions
  CheckSupportPoints();

  // build the kd tree
  BuildKDTree(pt.get<std::size_t>("MaxLeaf", 10));

  // find the required nearest neighbors for each support point
  FindNearestNeighbors();
}

void SupportPointCloud::FindNearestNeighbors() const {
  // loop through each support point
  for( const auto& point : supportPoints ) {
    // the maximum required nearest neighbors
    const std::size_t maxNeigh = *std::max_element(point->numNeighbors.begin(), point->numNeighbors.end());
    if( maxNeigh>supportPoints.size() ) { throw exceptions::SupportPointCloudNotEnoughPointsException(supportPoints.size(), maxNeigh); }

    // find the nearest nieghbors
    std::vector<std::size_t> neighInd; std::vector<double> neighDist;
    FindNearestNeighbors(point->x, maxNeigh, neighInd, neighDist);
    assert(neighInd.size()==maxNeigh); assert(neighDist.size()==maxNeigh);

    // set the nearest neighbors
    point->SetNearestNeighbors(neighInd, neighDist);
  }

  if( requireConnectedGraphs ) {
    // check to make sure the graph is connected
    for( std::size_t i=0; i<OutputDimension(); ++i ) { if( !CheckConnected(i) ) { throw exceptions::SupportPointCloudNotConnected(i); } }
  }
}

bool SupportPointCloud::CheckConnected(std::size_t const outdim) const {
  std::vector<bool> visited(supportPoints.size(), false);

  CheckConnected(outdim, 0, visited);

  for( const auto& node : visited ) { if( !node ) { return false; } }
  return true;
}

void SupportPointCloud::CheckConnected(std::size_t const outdim, std::size_t const ind, std::vector<bool>& visited) const {
  // mark the current node as visited
  assert(ind<visited.size());
  visited[ind] = true;

  // get the connected neighbors and recursively mark them as visited
  const std::vector<size_t> neighbors = supportPoints[ind]->GlobalNeighborIndices(outdim);
  for( std::size_t i=0; i<neighbors.size(); ++i ) {
    assert(neighbors[i]<visited.size());
    if( !visited[neighbors[i]] ) { CheckConnected(outdim, neighbors[i], visited); }
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
  neighInd.resize(k); neighDist.resize(k);

  // find the nearest neighbors
  const std::size_t nfound = kdtree->knnSearch(point.data(), k, neighInd.data(), neighDist.data());
  assert(nfound==k);
}

std::vector<std::shared_ptr<SupportPoint> >::const_iterator SupportPointCloud::Begin() const { return supportPoints.begin(); }

std::vector<std::shared_ptr<SupportPoint> >::const_iterator SupportPointCloud::End() const { return supportPoints.end(); }
