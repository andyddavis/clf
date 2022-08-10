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
    const std::string expected = "CLF Error: Domain::Inside not yet implemented";
    const std::string err = exc.what();
    EXPECT_TRUE(err==expected);
  }

  try { 
    dom.MapToHypercube(x);
  } catch( exceptions::NotImplemented const& exc ) {
    const std::string expected = "CLF Error: Domain::MapToHypercube not yet implemented";
    const std::string err = exc.what();
    EXPECT_TRUE(err==expected);
  }

  try { 
    dom.Sample();
  } catch( exceptions::NotImplemented const& exc ) {
    const std::string expected = "CLF Error: Domain::Sample not yet implemented";
    const std::string err = exc.what();
    EXPECT_TRUE(err==expected);
  }
}
