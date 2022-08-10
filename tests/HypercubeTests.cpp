#include <gtest/gtest.h>

#include "clf/Hypercube.hpp"

using namespace clf;

TEST(HypercubeTests, UnitCube) {
  const std::size_t dim = 4;

  auto para = std::make_shared<Parameters>();
  para->Add("InputDimension", dim);

  Hypercube dom0(dim);
  EXPECT_EQ(dom0.dim, dim);

  Hypercube dom1(para);
  EXPECT_EQ(dom1.dim, dim);

  for( std::size_t i=0; i<dim; ++i ) {
    EXPECT_DOUBLE_EQ(dom0.LeftBoundary(i), 0.0);
    EXPECT_DOUBLE_EQ(dom1.LeftBoundary(i), 0.0);
    EXPECT_DOUBLE_EQ(dom0.RightBoundary(i), 1.0);
    EXPECT_DOUBLE_EQ(dom1.RightBoundary(i), 1.0);
  }

  Eigen::VectorXd x = (Eigen::VectorXd::Ones(dim)+Eigen::VectorXd::Random(dim))/2.0;
  { // check inside
    EXPECT_TRUE(dom0.Inside(x));
    EXPECT_TRUE(dom1.Inside(x));
    const std::size_t check = rand()%dim;
    x(check) = 2.0;
    EXPECT_FALSE(dom0.Inside(x));
    EXPECT_FALSE(dom1.Inside(x));
    x(check) = -0.254;
    EXPECT_FALSE(dom0.Inside(x));
    EXPECT_FALSE(dom1.Inside(x));
    x(check) = 0.532;
    EXPECT_TRUE(dom0.Inside(x));
    EXPECT_TRUE(dom1.Inside(x));
  }

  const Eigen::VectorXd y0 = dom0.MapToHypercube(x);
  const Eigen::VectorXd y1 = dom1.MapToHypercube(x);
  const Hypercube cube(-1.0, 1.0, dim);
  EXPECT_TRUE(cube.Inside(y0));
  EXPECT_TRUE(cube.Inside(y1));

  const Eigen::VectorXd expected = 2.0*x-Eigen::VectorXd::Constant(dim, 1.0);
  EXPECT_NEAR((y0-expected).norm(), 0.0, 1.0e-13);
  EXPECT_NEAR((y1-expected).norm(), 0.0, 1.0e-13);

  for( std::size_t i=0; i<10; ++i ) {
    const Eigen::VectorXd samp0 = dom0.Sample();
    EXPECT_EQ(samp0.size(), dim);
    EXPECT_TRUE(dom0.Inside(samp0));
    const Eigen::VectorXd samp1 = dom1.Sample();
    EXPECT_EQ(samp1.size(), dim);
    EXPECT_TRUE(dom1.Inside(samp1));
  }
}

TEST(HypercubeTests, FixedCube) {
  const double left = -8.0;
  const double right = 4.0;
  const std::size_t dim = 4;

  auto para = std::make_shared<Parameters>();
  para->Add("InputDimension", dim);
  para->Add("LeftBoundary", left);
  para->Add("RightBoundary", right);

  Hypercube dom0(left, right, dim);
  EXPECT_EQ(dom0.dim, dim);

  Hypercube dom1(para);
  EXPECT_EQ(dom1.dim, dim);

  for( std::size_t i=0; i<dim; ++i ) {
    EXPECT_DOUBLE_EQ(dom0.LeftBoundary(i), left);
    EXPECT_DOUBLE_EQ(dom1.LeftBoundary(i), left);
    EXPECT_DOUBLE_EQ(dom0.RightBoundary(i), right);
    EXPECT_DOUBLE_EQ(dom1.RightBoundary(i), right);
  }
  
  Eigen::VectorXd x = Eigen::VectorXd::Constant(dim, -2.0)+6.0*Eigen::VectorXd::Random(dim);
  { // check inside
    EXPECT_TRUE(dom0.Inside(x));
    EXPECT_TRUE(dom1.Inside(x));
    const std::size_t check = rand()%dim;
    x(check) = 10.0;
    EXPECT_FALSE(dom0.Inside(x));
    EXPECT_FALSE(dom1.Inside(x));
    x(check) = -9.254;
    EXPECT_FALSE(dom0.Inside(x));
    EXPECT_FALSE(dom1.Inside(x));
    x(check) = 0.532;
    EXPECT_TRUE(dom0.Inside(x));
    EXPECT_TRUE(dom1.Inside(x));
  }

  const Eigen::VectorXd y0 = dom0.MapToHypercube(x);
  const Eigen::VectorXd y1 = dom1.MapToHypercube(x);
  const Hypercube cube(-1.0, 1.0, dim);
  EXPECT_TRUE(cube.Inside(y0));
  EXPECT_TRUE(cube.Inside(y1));

  const Eigen::VectorXd center = Eigen::VectorXd::Constant(dim, (left+right)/2.0);
  const Eigen::ArrayXd delta = Eigen::ArrayXd::Constant(dim, right-left);
  const Eigen::VectorXd expected = 2.0*(x-center).array()/delta;
  EXPECT_NEAR((y0-expected).norm(), 0.0, 1.0e-13);
  EXPECT_NEAR((y1-expected).norm(), 0.0, 1.0e-13);

  for( std::size_t i=0; i<10; ++i ) {
    const Eigen::VectorXd samp0 = dom0.Sample();
    EXPECT_EQ(samp0.size(), dim);
    EXPECT_TRUE(dom0.Inside(samp0));
    const Eigen::VectorXd samp1 = dom1.Sample();
    EXPECT_EQ(samp1.size(), dim);
    EXPECT_TRUE(dom1.Inside(samp1));
  }
}

TEST(HypercubeTests, RandomCube) {
  const std::size_t dim = 4;

  Eigen::VectorXd left = 0.1*Eigen::VectorXd::Random(dim);
  Eigen::VectorXd right = Eigen::VectorXd::Random(dim);
  for( std::size_t i=0; i<dim; ++i ) {
    if( right(i)<left(i) ) { std::swap(right(i), left(i)); }
  }

  Hypercube dom(left, right);
  EXPECT_EQ(dom.dim, dim);

  Eigen::VectorXd x = Eigen::VectorXd::Random(dim);
  for( std::size_t i=0; i<dim; ++i ) { x(i) = ( right(i)+left(i) + x(i)*(right(i)-left(i)) )/2.0; }
  { // check inside
    EXPECT_TRUE(dom.Inside(x));
    const std::size_t check = rand()%dim;
    x(check) = right(check) + 5.328;
    EXPECT_FALSE(dom.Inside(x));
    x(check) = left(check) - 8.4302;
    EXPECT_FALSE(dom.Inside(x));
    x(check) = (right(check)+left(check))/2.0;
    EXPECT_TRUE(dom.Inside(x));
  }

  Eigen::VectorXd y = dom.MapToHypercube(x);

  const Hypercube cube(-1.0, 1.0, dim);
  EXPECT_TRUE(cube.Inside(y));

  const Eigen::VectorXd center = (left+right)/2.0;
  const Eigen::ArrayXd delta = right-left;
  const Eigen::VectorXd expected = 2.0*(x-center).array()/delta;
  EXPECT_NEAR((y-expected).norm(), 0.0, 1.0e-13);

  for( std::size_t i=0; i<10; ++i ) {
    const Eigen::VectorXd samp = dom.Sample();
    EXPECT_EQ(samp.size(), dim);
    EXPECT_TRUE(dom.Inside(samp));
  }
}
