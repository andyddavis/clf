#include <gtest/gtest.h>

#include "clf/Hypercube.hpp"

#include "clf/PointCloud.hpp"

using namespace clf;

TEST(PointCloudTests, AddPoints) {
  const std::size_t dim = 3;

  PointCloud cloud; // create an empty point cloud
  EXPECT_EQ(cloud.NumPoints(), 0);

  // create point to add later (so the global id is less than points created after it)
  auto firstPt = std::make_shared<Point>(Eigen::VectorXd::Random(dim));

  // create a random point 
  auto pt1 = std::make_shared<Point>(Eigen::VectorXd::Random(dim));
  cloud.AddPoint(pt1);
  EXPECT_EQ(cloud.NumPoints(), 1);

  // add the same point, nothing should happen
  cloud.AddPoint(pt1);
  EXPECT_EQ(cloud.NumPoints(), 1);
  EXPECT_NEAR((cloud.GetUsingID(pt1->id)->x-pt1->x).norm(), 0.0, 1.0e-14);

  auto pt2 = std::make_shared<Point>(Eigen::VectorXd::Random(dim));
  cloud.AddPoint(pt2);
  EXPECT_EQ(cloud.NumPoints(), 2);
  EXPECT_NEAR((cloud.GetUsingID(pt1->id)->x-pt1->x).norm(), 0.0, 1.0e-14);
  EXPECT_NEAR((cloud.GetUsingID(pt2->id)->x-pt2->x).norm(), 0.0, 1.0e-14);

  // add the same point, nothing should happen
  cloud.AddPoint(pt1);
  cloud.AddPoint(pt2);
  EXPECT_EQ(cloud.NumPoints(), 2);

  // create a random point in place
  cloud.AddPoint(firstPt);
  EXPECT_EQ(cloud.NumPoints(), 3);
  EXPECT_NEAR((cloud.GetUsingID(pt1->id)->x-pt1->x).norm(), 0.0, 1.0e-14);
  EXPECT_NEAR((cloud.GetUsingID(pt2->id)->x-pt2->x).norm(), 0.0, 1.0e-14);
  EXPECT_NEAR((cloud.GetUsingID(firstPt->id)->x-firstPt->x).norm(), 0.0, 1.0e-14);

  // add the same point, nothing should happen
  cloud.AddPoint(pt1);
  cloud.AddPoint(pt2);
  cloud.AddPoint(firstPt);
  EXPECT_EQ(cloud.NumPoints(), 3);

  // the points should be ordered by their id
  std::vector<std::shared_ptr<Point> > points({pt1, pt2, firstPt});
  std::sort(points.begin(), points.end(), [](std::shared_ptr<Point> const& p1, std::shared_ptr<Point> const& p2) { return p1->id<p2->id; });
  for( std::size_t i=0; i<points.size(); ++i ) { EXPECT_NEAR((cloud.Get(i)->x-points[i]->x).norm(), 0.0, 1.0e-14); }
}

TEST(PointCloudTests, DomainCheck) {
  const std::size_t dim = 3; // the dimension
  const std::size_t n = 100; // the number of points
  
  auto domain = std::make_shared<Hypercube>(Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Ones(dim));

  PointCloud cloud(domain); // create an empty point cloud
  EXPECT_EQ(cloud.NumPoints(), 0);

  // add points by sampling the domain
  cloud.AddPoint();
  EXPECT_EQ(cloud.NumPoints(), 1);
  cloud.AddPoints(n-1);
  EXPECT_EQ(cloud.NumPoints(), n);

  // get the distance between the points
  for( std::size_t i=0; i<n; ++i ) {
    auto pi = cloud.Get(i);
    for( std::size_t j=0; j<n; ++j ) {
      auto pj = cloud.Get(j);
      const double expected = (pi->x-pj->x).norm();
      EXPECT_NEAR(cloud.Distance(i, j), expected, 1.0e-13);
    }
  }

  const std::size_t numNeighs = rand()%(n-1);

  for( std::size_t i=0; i<cloud.NumPoints(); ++i ) {
    // find the nearest neighbors
    std::vector<std::size_t> neighbors = cloud.NearestNeighbors(i, numNeighs);
    EXPECT_EQ(neighbors.size(), numNeighs);
    // the first neighbor is itself
    EXPECT_EQ(neighbors[0], i);
    EXPECT_NEAR(cloud.Distance(i, neighbors[0]), 0.0, 1.0e-13);
    // they shoud be increasing order
    for( std::size_t k=1; k<numNeighs; ++k ) {
      EXPECT_TRUE(cloud.Distance(i, neighbors[k-1])<cloud.Distance(i, neighbors[k]));
    }
    for( std::size_t j=0; j<cloud.NumPoints(); ++j ) {
      if( j==i ) { continue; }
      auto it = std::find(neighbors.begin(), neighbors.end(), j);
      if( it!=neighbors.end() ) { continue; }
      EXPECT_TRUE(cloud.Distance(i, j)>cloud.Distance(i, *(it-1)));
    }
  }

  // add a bunch more points
  cloud.AddPoints(n);
  EXPECT_EQ(cloud.NumPoints(), 2*n);

  for( std::size_t i=0; i<cloud.NumPoints(); ++i ) {
    // find the nearest neighbors
    std::vector<std::size_t> neighbors = cloud.NearestNeighbors(i, numNeighs);
    EXPECT_EQ(neighbors.size(), numNeighs);
    // the first neighbor is itself
    EXPECT_EQ(neighbors[0], i);
    EXPECT_NEAR(cloud.Distance(i, neighbors[0]), 0.0, 1.0e-13);
    // they shoud be increasing order
    for( std::size_t k=1; k<numNeighs; ++k ) {
      EXPECT_NE(neighbors[k], i);
      EXPECT_TRUE(cloud.Distance(i, neighbors[k-1])<cloud.Distance(i, neighbors[k]));
    }
    for( std::size_t j=0; j<cloud.NumPoints(); ++j ) {
      if( j==i ) { continue; }
      auto it = std::find(neighbors.begin(), neighbors.end(), j);
      if( it!=neighbors.end() ) { continue; }
      EXPECT_TRUE(cloud.Distance(i, j)>cloud.Distance(i, *(it-1)));
    }
  }

  const Eigen::VectorXd x = Eigen::VectorXd::Random(dim);
  const std::pair<std::size_t, double> nearest = cloud.ClosestPoint(x);
  for( std::size_t j=0; j<cloud.NumPoints(); ++j ) {
    if( j==nearest.first) { continue; }
    EXPECT_TRUE(domain->Distance(x, cloud.Get(j)->x)>nearest.second);
  }
}
