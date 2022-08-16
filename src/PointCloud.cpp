#include "clf/PointCloud.hpp"

using namespace clf;

PointCloud::PointCloud(std::shared_ptr<Domain> const& domain) :
  domain(domain)
{}

std::size_t PointCloud::NumPoints() const { return points.size(); }

void PointCloud::AddPoint(std::shared_ptr<Point> const& point) { 
  if( points.size()==0 ) {
    points.push_back(point);
    return;
  }

  assert(point->x.size()==points[0]->x.size());

  auto it = std::upper_bound(points.begin(), points.end(), point, [](std::shared_ptr<Point> const& p1, std::shared_ptr<Point> const& p2) { return p1->id<=p2->id; });

  if( it==points.end() ) {
    points.push_back(point);
  } else if( (*it)->id!=point->id ) {
    points.insert(it, point);
  }
}

void PointCloud::AddPoint() {
  assert(domain);
  AddPoint(std::make_shared<Point>(domain->Sample()));  
}

void PointCloud::AddPoints(std::size_t const n) { for( std::size_t i=0; i<n; ++i ) { AddPoint(); } }

std::shared_ptr<Point> PointCloud::Get(std::size_t const ind) const { 
  assert(ind<points.size()); 
  return points[ind];
}

std::size_t PointCloud::PairHash::operator()(std::pair<std::size_t, std::size_t> const& p) const {
  auto h1 = std::hash<std::size_t>{}(p.first);
  auto h2 = std::hash<std::size_t>{}(p.second);
  
  return h1 ^ h2;  
}

double PointCloud::Distance(std::size_t const ind1, std::size_t const ind2) const {
  // it is the same point
  if( ind1==ind2 ) { return 0.0; }
  
  const std::pair<std::size_t, std::size_t> key = std::pair<std::size_t, std::size_t>(std::min(points[ind1]->id, points[ind2]->id), std::max(points[ind1]->id, points[ind2]->id));
  auto it = distances.find(key);
  
  // we have already computed the distance
  if( it!=distances.end() ) { return it->second; }

  // compute the distance
  assert(domain);
  const double dist = domain->Distance(points[ind1]->x, points[ind2]->x);

  // add the distance and update the nearest neighbors
  distances[key] = dist;
  UpdateNeighbors(key.first, key.second, dist);
  UpdateNeighbors(key.second, key.first, dist);
  
  return dist;
}

std::size_t PointCloud::IndexFromID(std::size_t const id) const {
  auto it = std::upper_bound(points.begin(), points.end(), id, [](std::size_t const id, std::shared_ptr<Point> const& p) { return id<=p->id; });
  assert(it!=points.end());
  return it-points.begin();
}

std::shared_ptr<Point> PointCloud::GetUsingID(std::size_t const id) const { return points[IndexFromID(id)]; }

void PointCloud::UpdateNeighbors(std::size_t const id, std::size_t const neigh, double const dist) const {
  // get the neighbor list for this index
  auto it = neighbors.find(id);
  if( it==neighbors.end() ) {
    bool check;
    std::tie(it, check) = neighbors.insert({id, std::vector<std::size_t>(1, id)});
    assert(check);
  }
  assert(it!=neighbors.end());

  // find the first index that is farther away
  const std::size_t ind = IndexFromID(id);
  auto jt = std::upper_bound(it->second.begin(), it->second.end(), dist, [this, &ind](double const dist, std::size_t const n) { return dist<Distance(ind, IndexFromID(n)); });

  // if we are not at the end
  if( jt==it->second.end() || (*jt)!=neigh ) { it->second.insert(jt, neigh); }
}

std::vector<std::size_t> PointCloud::NearestNeighbors(std::size_t const ind, std::size_t const k) const {
  if( k==0 ) { return std::vector<std::size_t>(); }
  assert(k<points.size());
  
  // find the neighbors of this key
  auto it = neighbors.find(points[ind]->id);

  // we have not computed the neighbors or points have been added since the last time we did
  if( it==neighbors.end() || it->second.size()!=NumPoints() ) {
    // compute the distance between this point and all of the other points---distance function updates neighbors
    for( std::size_t i=0; i<points.size(); ++i ) { Distance(ind, i); }
  }

  assert(it!=neighbors.end());

  // get the k nearest neighbors
  std::vector<std::size_t> neighs(k);
  for( std::size_t i=0; i<k; ++i ) { neighs[i] = IndexFromID(it->second[i]); }
  return neighs;
}

std::pair<std::size_t, double> PointCloud::ClosestPoint(Eigen::VectorXd const& x) const {
  assert(domain);
  
  double dist = std::numeric_limits<double>::infinity();
  std::size_t ind;
  for( std::size_t i=0; i<points.size(); ++i ) {
    const double d = domain->Distance(x, points[i]->x);
    if( d<dist ) { dist = d; ind = i; }
  }

  return std::pair<std::size_t, double>(ind, dist);
}
