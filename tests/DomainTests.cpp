#include <gtest/gtest.h>

#include "clf/CLFExceptions.hpp"

#include "clf/Domain.hpp"

using namespace clf;

TEST(DomainTests, DefaultImplementations) {
  const std::size_t dim = 4;
  
  Domain dom(dim);
  EXPECT_EQ(dom.dim, dim);

  const Eigen::VectorXd x = Eigen::VectorXd::Random(dim);

  try { 
    dom.Inside(x);
  } catch( exceptions::NotImplemented const& exc ) {
    const std::string expected = "CLF Error: Domain::CheckInside not yet implemented";
    const std::string err = exc.what();
    EXPECT_TRUE(err==expected);
  }

  const Eigen::VectorXd y = dom.MapToHypercube(x);
  EXPECT_NEAR((x-y).norm(), 0.0, 1.0e-13);

  try { 
    dom.Sample();
  } catch( exceptions::NotImplemented const& exc ) {
    const std::string expected = "CLF Error: Domain::ProposeSample not yet implemented";
    const std::string err = exc.what();
    EXPECT_TRUE(err==expected);
  }

  const Eigen::VectorXd x1 = Eigen::VectorXd::Random(dim);
  const Eigen::VectorXd x2 = Eigen::VectorXd::Random(dim);
  EXPECT_NEAR(dom.Distance(x1, x2), (x1-x2).norm(), 1.0e-13);
}

