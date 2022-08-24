#include <gtest/gtest.h>

#include "clf/Hypercube.hpp"

namespace clf {
namespace tests {

/// A class to run the tests for clf::Hypercube
class HypercubeTests : public::testing::Test {
protected:

  /// Check that the clf::Hypercube is correct
  /**
     @param[in] dom The domain we are checking
     @param[in] left The left boundary for each dimension
     @param[in] right The right boundary for each dimension
   */
  void CheckDomain(std::shared_ptr<Hypercube> const& dom, Eigen::VectorXd const& left, Eigen::VectorXd const& right) const {
    EXPECT_EQ(dom->dim, dim);

    for( std::size_t i=0; i<dim; ++i ) {
      EXPECT_DOUBLE_EQ(dom->LeftBoundary(i), left(i));
      EXPECT_DOUBLE_EQ(dom->RightBoundary(i), right(i));
    }

    Eigen::VectorXd x(dim);
    for( std::size_t i=0; i<dim; ++i ) { x(i) = left(i) + (right(i)-left(i))*rand()/(double)RAND_MAX; }
    const std::size_t check = rand()%(dim-1);
    // check inside
    EXPECT_TRUE(dom->Inside(x));
    x(check) = right(check)+10.0;;
    EXPECT_FALSE(dom->Inside(x));
    x(check) = left(check)-28.087;;
    EXPECT_FALSE(dom->Inside(x));
    x(check) = (left(check)+right(check))/2.0;
    EXPECT_TRUE(dom->Inside(x));

    const Eigen::VectorXd y = dom->MapToHypercube(x);
    const Hypercube cube(-1.0, 1.0, dim);
    EXPECT_TRUE(cube.Inside(y));
    
    const Eigen::VectorXd center = (left+right)/2.0;
    const Eigen::ArrayXd delta = right-left;
    const Eigen::VectorXd expected = 2.0*(x-center).array()/delta;
    EXPECT_NEAR((y-expected).norm(), 0.0, 1.0e-13);
    
    for( std::size_t i=0; i<10; ++i ) {
      const Eigen::VectorXd samp = dom->Sample();
      EXPECT_EQ(samp.size(), dim);
      EXPECT_TRUE(dom->Inside(samp));

      const std::pair<Eigen::VectorXd, Eigen::VectorXd> sampBoundary = dom->SampleBoundary();
      EXPECT_EQ(sampBoundary.first.size(), dim);
      EXPECT_EQ(sampBoundary.second.size(), dim);
      EXPECT_TRUE(dom->Inside(sampBoundary.first));
      EXPECT_NEAR(std::min((sampBoundary.first-left).array().abs().minCoeff(), (sampBoundary.first-right).array().abs().minCoeff()), 0.0, 1.0e-14);
      EXPECT_NEAR(sampBoundary.second.norm(), 1.0, 1.0e-14);
    }
  }

  /// Check that the periodic implementation of clf::Hypercube is correct
  /**
     @param[in] dom The domain we are checking
     @param[in] left The left boundary for each dimension
     @param[in] right The right boundary for each dimension
     @param[in] periodic Which boundaries are periodic?
   */
  void CheckDomain(std::shared_ptr<Hypercube> const& dom, Eigen::VectorXd const& left, Eigen::VectorXd const& right, std::vector<bool> const& periodic) const {
    EXPECT_EQ(dom->dim, dim);

    for( std::size_t i=0; i<dim; ++i ) {
      EXPECT_DOUBLE_EQ(dom->LeftBoundary(i), left(i));
      EXPECT_DOUBLE_EQ(dom->RightBoundary(i), right(i));
    }

    // this check only works is something is periodic
    assert(std::find(periodic.begin(), periodic.end(), true)!=periodic.end());

    // choose a random point that does not wrap around
    Eigen::VectorXd x1(dim), x2(dim);
    for( std::size_t i=0; i<dim; ++i ) {
      const double length = right(i)-left(i); assert(length>0.0);
      x1(i) = left(i) + length*(0.25 + 0.5*rand()/(double)RAND_MAX);
      x2(i) = left(i) + length*(0.25 + 0.5*rand()/(double)RAND_MAX);
    }
    EXPECT_TRUE(dom->Inside(x1));
    EXPECT_TRUE(dom->Inside(x2));
    double dist = dom->Distance(x1, x2);
    EXPECT_NEAR(dist, (x1-x2).norm(), 1.0e-14);

    // choose a point that does wrap around
    std::size_t check = rand()%dim;
    while( !periodic[check] ) { check = rand()%dim; }
    const double length = right(check)-left(check); assert(length>0.0);
    x1(check) = left(check)+length/5.0; x2(check) = right(check)-length/4.25;
    EXPECT_TRUE(dom->Inside(x1));
    EXPECT_TRUE(dom->Inside(x2));
    
    dist = dom->Distance(x1, x2);
    EXPECT_TRUE(dist<(x1-x2).norm());
    double expected = 0.0;
    for( std::size_t i=0; i<dim; ++i ) {
      if( periodic[i] ) {
	const double mn = std::min(x1(i), x2(i)), mx = std::max(x1(i), x2(i));
	const double diff = std::min(mx-mn, dom->RightBoundary(i)-mx+mn-dom->LeftBoundary(i));
	expected += diff*diff;
      } else {
	const double diff = x1(i)-x2(i);
	expected += diff*diff;
      }
    }
    EXPECT_NEAR(dist, std::sqrt(expected), 1.0e-14);

    // map outside the domain
    x1(check) = left(check)-length/5.0;
    EXPECT_TRUE(dom->Inside(x1));
    const std::optional<Eigen::VectorXd> y = dom->MapPeriodic(x1);
    EXPECT_TRUE(y);
    auto checkDom = std::make_shared<Hypercube>(left, right);
    EXPECT_FALSE(checkDom->Inside(x1));
    EXPECT_TRUE(checkDom->Inside(*y));

    for( std::size_t i=0; i<10; ++i ) {
      const Eigen::VectorXd x = dom->Sample();
      EXPECT_EQ(x.size(), dim);
      EXPECT_TRUE(dom->Inside(x));


      try {
	const std::pair<Eigen::VectorXd, Eigen::VectorXd> sampBoundary = dom->SampleBoundary();
	
	EXPECT_EQ(sampBoundary.first.size(), dim);
	EXPECT_EQ(sampBoundary.second.size(), dim);
	EXPECT_TRUE(dom->Inside(sampBoundary.first));
	EXPECT_NEAR(std::min((sampBoundary.first-left).array().abs().minCoeff(), (sampBoundary.first-right).array().abs().minCoeff()), 0.0, 1.0e-14);
	EXPECT_NEAR(sampBoundary.second.norm(), 1.0, 1.0e-14);
      } catch( Domain::SampleFailure const& exc ) {
	const std::string expected = "CLF Error: Tried to sample from the boundary of a clf::Hypercube but all coordinate directions are periodic. There is no boundary.";
	const std::string err = exc.what();
	EXPECT_TRUE(err==expected);
	EXPECT_TRUE(std::find(periodic.begin(), periodic.end(), false)==periodic.end());
	EXPECT_TRUE(dom->Periodic());
      }
    }
  }

  /// The parameters that define the domain
  std::shared_ptr<Parameters> para = std::make_shared<Parameters>();

  /// The dimension of the domain
  const std::size_t dim = 4;
};

TEST_F(HypercubeTests, UnitCube) {
  auto dom0 = std::make_shared<Hypercube>(dim);
  CheckDomain(dom0, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Ones(dim));

  para->Add("InputDimension", dim);
  auto dom1 = std::make_shared<Hypercube>(para);
  CheckDomain(dom1, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Ones(dim));
}

TEST_F(HypercubeTests, UnitCubePeriodic) {
  std::vector<bool> periodic(dim, true);
  auto dom0 = std::make_shared<Hypercube>(periodic);
  CheckDomain(dom0, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Ones(dim), periodic);
  
  for( std::size_t i=0; i<dim-1; ++i ) { periodic[i] = false; }
  para->Add("InputDimension", dim);
  auto dom1 = std::make_shared<Hypercube>(periodic, para);
  CheckDomain(dom1, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Ones(dim), periodic);
}
  
TEST_F(HypercubeTests, FixedCube) {
  const double left = -8.0;
  const double right = 4.0;

  auto dom0 = std::make_shared<Hypercube>(left, right, dim);
  CheckDomain(dom0, Eigen::VectorXd::Constant(dim, left), Eigen::VectorXd::Constant(dim, right));

  para->Add("InputDimension", dim);
  para->Add("LeftBoundary", left);
  para->Add("RightBoundary", right);
  auto dom1 = std::make_shared<Hypercube>(para);
  CheckDomain(dom1, Eigen::VectorXd::Constant(dim, left), Eigen::VectorXd::Constant(dim, right));
}

TEST_F(HypercubeTests, FixedCubePeriodic) {
  const double left = -8.0;
  const double right = 4.0;

  std::vector<bool> periodic(dim, true);
  auto dom0 = std::make_shared<Hypercube>(periodic, left, right);
  CheckDomain(dom0, Eigen::VectorXd::Constant(dim, left), Eigen::VectorXd::Constant(dim, right), periodic);

  for( std::size_t i=1; i<dim-1; ++i ) { periodic[i] = false; }
  para->Add("InputDimension", dim);
  para->Add("LeftBoundary", left);
  para->Add("RightBoundary", right);
  auto dom1 = std::make_shared<Hypercube>(periodic, para);
  CheckDomain(dom1, Eigen::VectorXd::Constant(dim, left), Eigen::VectorXd::Constant(dim, right), periodic);
}

TEST_F(HypercubeTests, RandomCube) {
  Eigen::VectorXd left = 0.1*Eigen::VectorXd::Random(dim);
  Eigen::VectorXd right = Eigen::VectorXd::Random(dim);
  for( std::size_t i=0; i<dim; ++i ) {
    if( right(i)<left(i) ) { std::swap(right(i), left(i)); }
  }

  auto dom = std::make_shared<Hypercube>(left, right);
  CheckDomain(dom, left, right);
}

TEST_F(HypercubeTests, RandomCubePeriodic) {
  Eigen::VectorXd left = 0.1*Eigen::VectorXd::Random(dim);
  Eigen::VectorXd right = Eigen::VectorXd::Random(dim);
  for( std::size_t i=0; i<dim; ++i ) {
    if( right(i)<left(i) ) { std::swap(right(i), left(i)); }
  }

  std::vector<bool> periodic(dim);
  for( std::size_t i=0; i<dim; ++i ) { periodic[i] = (rand()%2==0); }
  if( std::find(periodic.begin(), periodic.end(), true)==periodic.end() ) { periodic[0] = true; }

  auto dom = std::make_shared<Hypercube>(left, right, periodic);
  CheckDomain(dom, left, right, periodic);
}

TEST_F(HypercubeTests, Superset) {
  const double left0 = -8.0;
  const double right0 = 4.0;
  const double left1 = -4.0;
  const double right1 = 8.0;

  auto dom = std::make_shared<Hypercube>(left0, right0, dim);
  CheckDomain(dom, Eigen::VectorXd::Constant(dim, left0), Eigen::VectorXd::Constant(dim, right0));

  // make the super set
  auto super = std::make_shared<Hypercube>(left1, right1, dim);
  CheckDomain(super, Eigen::VectorXd::Constant(dim, left1), Eigen::VectorXd::Constant(dim, right1));

  // set the super set
  dom->SetSuperset(super);

  std::size_t check = rand()%(dim-1);

  Eigen::VectorXd x = Eigen::VectorXd::Zero(dim);
  x(check) = -5.0;
  EXPECT_FALSE(super->Inside(x));
  EXPECT_FALSE(dom->Inside(x));
  x(check) = 0.0;
  EXPECT_TRUE(super->Inside(x));
  EXPECT_TRUE(dom->Inside(x));

  for( std::size_t i=0; i<10; ++i ) {
    x = dom->Sample();
    EXPECT_TRUE(super->Inside(x));
    EXPECT_TRUE(dom->Inside(x));

    const std::pair<Eigen::VectorXd, Eigen::VectorXd> sampBoundary = dom->SampleBoundary();
    EXPECT_EQ(sampBoundary.first.size(), dim);
    EXPECT_TRUE(dom->Inside(sampBoundary.first));
    EXPECT_NEAR(std::min((sampBoundary.first-Eigen::VectorXd::Constant(dim, left0)).array().abs().minCoeff(), (sampBoundary.first-Eigen::VectorXd::Constant(dim, right0)).array().abs().minCoeff()), 0.0, 1.0e-14);
    EXPECT_TRUE(super->Inside(sampBoundary.first));
  }

  auto badDomain = std::make_shared<Hypercube>(10.0, 20.0, dim);
  CheckDomain(badDomain, Eigen::VectorXd::Constant(dim, 10.0), Eigen::VectorXd::Constant(dim, 20.0));
  badDomain->SetSuperset(super);

  for( std::size_t i=0; i<10; ++i ) {
    try {
      x = badDomain->Sample();
    } catch( Domain::SampleFailure const& exc ) {
      const std::string expected = "CLF Error: Domain::Sample did not propose a valid sample in 10000 proposals.";
      const std::string err = exc.what();
      EXPECT_TRUE(err==expected);
    }
  }
}

TEST_F(HypercubeTests, PeriodicSuperset) {
  const double left0 = -8.0;
  const double right0 = 3.0;
  const double left1 = -4.0;
  const double right1 = 8.0;

  auto dom = std::make_shared<Hypercube>(left0, right0, dim);
  CheckDomain(dom, Eigen::VectorXd::Constant(dim, left0), Eigen::VectorXd::Constant(dim, right0));

  // make the periodic super set
  std::vector<bool> periodic(dim);
  for( std::size_t i=0; i<dim; ++i ) { periodic[i] = (rand()%2==0); }
  if( std::find(periodic.begin(), periodic.end(), true)==periodic.end() ) { periodic[0] = true; }
  if( std::find(periodic.begin(), periodic.end(), false)==periodic.end() ) { periodic[dim-1] = false; }

  auto super = std::make_shared<Hypercube>(periodic, left1, right1);
  CheckDomain(super, Eigen::VectorXd::Constant(dim, left1), Eigen::VectorXd::Constant(dim, right1), periodic);

  // set the super set
  dom->SetSuperset(super);

  Eigen::VectorXd x = Eigen::VectorXd::Zero(dim);
  std::size_t check = rand()%dim;

  // check a non-periodic coordinate
  while( periodic[check] ) { check = rand()%dim; }
  x(check) = -5.0;
  EXPECT_FALSE(super->Inside(x));
  EXPECT_FALSE(dom->Inside(x));
  std::optional<Eigen::VectorXd> y = dom->MapPeriodic(x);
  EXPECT_TRUE(y);

  EXPECT_NEAR((*y)(check), -5.0, 1.0e-14);
  x(check) = 0.0;
  EXPECT_TRUE(super->Inside(x));
  EXPECT_TRUE(dom->Inside(x));
  y = dom->MapPeriodic(x);
  EXPECT_TRUE(y);
  EXPECT_NEAR((*y)(check), 0.0, 1.0e-14);

  // check a periodic coordinate
  while( !periodic[check] ) { check = rand()%dim; }
  x(check) = -5.0;
  EXPECT_TRUE(super->Inside(x));
  EXPECT_TRUE(dom->Inside(x));
  y = dom->MapPeriodic(x);
  EXPECT_NEAR((*y)(check), -5.0, 1.0e-14);

  x(check) = -8.5;
  EXPECT_TRUE(super->Inside(x));
  EXPECT_FALSE(dom->Inside(x));
  y = dom->MapPeriodic(x);
  // -8.5 + 12 (super set domain length) = 3.5 either option is kind of the same
  EXPECT_TRUE(std::abs((*y)(check)+8.5)<1.0e-14 || std::abs((*y)(check)-3.5)<1.0e-14);

  x(check) = -8.25;
  EXPECT_TRUE(super->Inside(x));
  EXPECT_FALSE(dom->Inside(x));
  y = dom->MapPeriodic(x);
  EXPECT_NEAR((*y)(check), -8.25, 1.0e-14);

  x(check) = -8.75;
  EXPECT_TRUE(super->Inside(x));
  EXPECT_FALSE(dom->Inside(x));
  y = dom->MapPeriodic(x);
  EXPECT_NEAR((*y)(check), 3.25, 1.0e-14);

  x(check) = -20.5;
  EXPECT_TRUE(super->Inside(x));
  EXPECT_FALSE(dom->Inside(x));
  y = dom->MapPeriodic(x);
  // -8.5 + 12 (super set domain length) = 3.5 either option is kind of the same
  EXPECT_TRUE(std::abs((*y)(check)+8.5)<1.0e-14 || std::abs((*y)(check)-3.5)<1.0e-14);

  x(check) = -9.5;
  EXPECT_TRUE(super->Inside(x));
  EXPECT_TRUE(dom->Inside(x));
  y = dom->MapPeriodic(x);
  EXPECT_NEAR((*y)(check), 2.5, 1.0e-14);
  
  x(check) = 14.5;
  EXPECT_TRUE(super->Inside(x));
  EXPECT_TRUE(dom->Inside(x));
  y = dom->MapPeriodic(x);
  EXPECT_NEAR((*y)(check), 2.5, 1.0e-14);
  
  x(check) = 3.5;
  EXPECT_TRUE(super->Inside(x));
  EXPECT_FALSE(dom->Inside(x));
  y = dom->MapPeriodic(x);
  // -8.5 + 12 (super set domain length) = 3.5 either option is kind of the same
  EXPECT_TRUE(std::abs((*y)(check)+8.5)<1.0e-14 || std::abs((*y)(check)-3.5)<1.0e-14);

  x(check) = 27.5;
  EXPECT_TRUE(super->Inside(x));
  EXPECT_FALSE(dom->Inside(x));
  y = dom->MapPeriodic(x);
  // -8.5 + 12 (super set domain length) = 3.5 either option is kind of the same
  EXPECT_TRUE(std::abs((*y)(check)+8.5)<1.0e-14 || std::abs((*y)(check)-3.5)<1.0e-14);

  x(check) = 3.25;
  EXPECT_TRUE(super->Inside(x));
  EXPECT_FALSE(dom->Inside(x));
  y = dom->MapPeriodic(x);
  EXPECT_NEAR((*y)(check), 3.25, 1.0e-14);

  x(check) = 3.75;
  EXPECT_TRUE(super->Inside(x));
  EXPECT_FALSE(dom->Inside(x));
  y = dom->MapPeriodic(x);
  EXPECT_NEAR((*y)(check), -8.25, 1.0e-14);

  x(check) = 15.75;
  EXPECT_TRUE(super->Inside(x));
  EXPECT_FALSE(dom->Inside(x));
  y = dom->MapPeriodic(x);
  EXPECT_NEAR((*y)(check), -8.25, 1.0e-14);

  x(check) = 0.0;
  EXPECT_TRUE(super->Inside(x));
  EXPECT_TRUE(dom->Inside(x));

  auto superNonPeriodic = std::make_shared<Hypercube>(left1, right1, dim);
  CheckDomain(superNonPeriodic, Eigen::VectorXd::Constant(dim, left1), Eigen::VectorXd::Constant(dim, right1));
  for( std::size_t i=0; i<25; ++i ) {
    x = dom->Sample();
    EXPECT_TRUE(super->Inside(x));
    EXPECT_TRUE(superNonPeriodic->Inside(x));
    EXPECT_TRUE(dom->Inside(x));

    const std::pair<Eigen::VectorXd, Eigen::VectorXd> sampBoundary = dom->SampleBoundary();
    EXPECT_EQ(sampBoundary.first.size(), dim);
    EXPECT_TRUE(dom->Inside(sampBoundary.first));
    const Eigen::VectorXd mappedBoundary = dom->MapToHypercube(sampBoundary.first);
    EXPECT_NEAR(std::min((mappedBoundary+Eigen::VectorXd::Ones(dim)).array().abs().minCoeff(), (mappedBoundary-Eigen::VectorXd::Ones(dim)).array().abs().minCoeff()), 0.0, 1.0e-14);
    EXPECT_TRUE(super->Inside(sampBoundary.first));
    EXPECT_TRUE(superNonPeriodic->Inside(x));
  }
}
  
} // namespace tests
} // namespace clf
