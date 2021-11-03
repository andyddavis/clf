#include "clf/SupportPointCloud.hpp"

namespace pt = boost::property_tree;
using namespace clf;

SupportPointCloud::SupportPointCloud(std::vector<std::shared_ptr<SupportPoint> > const& supportPoints, pt::ptree const& pt) :
PointCloud(std::vector<std::shared_ptr<Point> >(supportPoints.begin(), supportPoints.end())),
//supportPoints(supportPoints),
numCoefficients(NumCoefficients(supportPoints)),
requireConnectedGraphs(pt.get<bool>("RequireConnectedGraphs", false))
{
  // make sure the support points all of the same in/output dimensions
  CheckSupportPoints();

  // build the kd tree
  BuildKDTree(pt.get<std::size_t>("MaxLeaf", 10));
}

std::size_t SupportPointCloud::NumCoefficients(std::vector<std::shared_ptr<SupportPoint> > const& supportPoints) {
  std::size_t numCoeffs = 0;
  for( const auto& it : supportPoints ) { numCoeffs += it->NumCoefficients(); }
  return numCoeffs;
}

std::shared_ptr<SupportPointCloud> SupportPointCloud::Construct(std::shared_ptr<SupportPointSampler> const& sampler, boost::property_tree::ptree const& pt) {
  std::vector<std::shared_ptr<SupportPoint> > points(pt.get<std::size_t>("NumSupportPoints"));
  for( std::size_t i=0; i<points.size(); ++i ) { points[i] = sampler->Sample(); }

  return Construct(points, pt);
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
  for( const auto& pnt : points ) {
    auto point = std::dynamic_pointer_cast<SupportPoint>(pnt);
    assert(point);
    // the maximum required nearest neighbors
    const std::size_t numNeigh = point->NumNeighbors();
    if( numNeigh>points.size() ) { throw exceptions::SupportPointCloudNotEnoughPointsException(points.size(), numNeigh); }

    // find the nearest nieghbors
    std::vector<unsigned int> neighInd; std::vector<double> neighDist;
    FindNearestNeighbors(point->x, numNeigh, neighInd, neighDist);
    assert(neighInd.size()==numNeigh); assert(neighDist.size()==numNeigh);

    // set the nearest neighbors
    point->SetNearestNeighbors(shared_from_this(), neighInd, neighDist);
  }

  // now that all of the points know their neighbors, we can create the coupling costs
  for( const auto& point : points ) { std::dynamic_pointer_cast<SupportPoint>(point)->CreateCoupledCosts(); }

  if( requireConnectedGraphs ) {
    // check to make sure the graph is connected
    if( !CheckConnected() ) { throw exceptions::SupportPointCloudNotConnected(); }
  }
}

bool SupportPointCloud::CheckConnected() const {
  std::vector<bool> visited(points.size(), false);

  CheckConnected(0, visited);

  for( const auto& node : visited ) { if( !node ) { return false; } }
  return true;
}

void SupportPointCloud::CheckConnected(std::size_t const ind, std::vector<bool>& visited) const {
  // mark the current node as visited
  assert(ind<visited.size());
  visited[ind] = true;

  // get the connected neighbors and recursively mark them as visited
  const std::vector<size_t> neighbors = GetSupportPoint(ind)->GlobalNeighborIndices();
  for( std::size_t i=0; i<neighbors.size(); ++i ) {
    assert(neighbors[i]<visited.size());
    if( !visited[neighbors[i]] ) { CheckConnected(neighbors[i], visited); }
  }
}

void SupportPointCloud::CheckSupportPoints() const {
  for( std::size_t i=1; i<points.size(); ++i ) {
    // check the input dimension
    if( GetSupportPoint(i-1)->model->inputDimension!=GetSupportPoint(i)->model->inputDimension ) { throw exceptions::SupportPointCloudDimensionException(exceptions::SupportPointCloudDimensionException::Type::INPUT, i-1, i); }

    // check the output dimension
    if( GetSupportPoint(i-1)->model->outputDimension!=GetSupportPoint(i)->model->outputDimension ) { throw exceptions::SupportPointCloudDimensionException(exceptions::SupportPointCloudDimensionException::Type::OUTPUT, i-1, i); }
  }
}

void SupportPointCloud::BuildKDTree(std::size_t const maxLeaf) {
  kdtree = std::make_shared<NanoflannKDTree>(InputDimension(), *this, nanoflann::KDTreeSingleIndexAdaptorParams(maxLeaf));
  kdtree->buildIndex();
}

std::shared_ptr<SupportPoint> SupportPointCloud::GetSupportPoint(std::size_t const i) const {
  auto point = std::dynamic_pointer_cast<SupportPoint>(points[i]);
  assert(point);
  return point;
}

std::size_t SupportPointCloud::kdtree_get_point_count() const { return NumPoints(); }

double SupportPointCloud::kdtree_get_pt(std::size_t const p, std::size_t const i) const {
  assert(p<points.size());
  return GetSupportPoint(p)->x(i);
}

std::size_t SupportPointCloud::InputDimension() const {
  assert(points.size()>0);
  return GetSupportPoint(0)->model->inputDimension;
}

std::size_t SupportPointCloud::OutputDimension() const {
  assert(points.size()>0);
  return GetSupportPoint(0)->model->outputDimension;
}

std::shared_ptr<SupportPoint> SupportPointCloud::NearestSupportPoint(Eigen::VectorXd const& point) const {
  std::vector<unsigned int> neighInd;
  std::vector<double> neighDist;
  FindNearestNeighbors(point, 1, neighInd, neighDist);
  assert(neighInd.size()==1); assert(neighDist.size()==1);

  return GetSupportPoint(neighInd[0]);
}

void SupportPointCloud::FindNearestNeighbors(Eigen::VectorXd const& point, std::size_t const k, std::vector<unsigned int>& neighInd, std::vector<double>& neighDist) const {
  assert(k<=points.size());

  // resize the input vecotrs
  assert(k>0);
  neighInd.resize(k); neighDist.resize(k);

  // find the nearest neighbors
  assert(kdtree);
  const std::size_t nfound = kdtree->knnSearch(point.data(), k, neighInd.data(), neighDist.data());
  assert(nfound==k);
}

std::pair<std::vector<unsigned int>, std::vector<double> > SupportPointCloud::FindNearestNeighbors(Eigen::VectorXd const& point, std::size_t const k) const {
  std::pair<std::vector<unsigned int>, std::vector<double> > result;
  SupportPointCloud::FindNearestNeighbors(point, k, result.first, result.second);
  return result;
}

Eigen::VectorXd SupportPointCloud::GetCoefficients() const {
  Eigen::VectorXd coeffs(numCoefficients);

  std::size_t ind = 0;
  for( const auto& pnt : points ) {
    auto point = std::dynamic_pointer_cast<SupportPoint>(pnt);
    assert(point);
    coeffs.segment(ind, point->NumCoefficients()) = point->Coefficients();
    ind += point->NumCoefficients();
  }

  return coeffs;
}

void SupportPointCloud::SetCoefficients(Eigen::VectorXd const& coeffs) {
  std::size_t ind = 0;
  for( const auto& pnt : points ) {
    auto point = std::dynamic_pointer_cast<SupportPoint>(pnt);
    assert(point);
    point->Coefficients() = coeffs.segment(ind, point->NumCoefficients());
    ind += point->NumCoefficients();
  }
}
