#include <gtest/gtest.h>

#include "clf/Hypercube.hpp"
#include "clf/LegendrePolynomials.hpp"

#include "clf/BurgersEquation.hpp"

using namespace clf;

namespace clf {
namespace tests {

/// A class to run the tests for clf::BurgersEquation
class BurgersEquationTests : public::testing::Test {
protected:

  /// Tear down the tests
  virtual void TearDown() override {
    EXPECT_EQ(system->indim, indim);
    EXPECT_EQ(system->outdim, 1);
    
    const std::size_t maxOrder = 4;
    std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);
    
    const double delta = 1.5;
    const Eigen::VectorXd xbar = Eigen::VectorXd::Random(indim);
    auto domain = std::make_shared<Hypercube>(xbar-Eigen::VectorXd::Constant(indim, delta), xbar+Eigen::VectorXd::Constant(indim, delta));
    
    const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);
    
    auto basis = std::make_shared<LegendrePolynomials>();
    auto func = std::make_shared<LocalFunction>(set, basis, domain, 1);
    
    const Eigen::VectorXd coeff = Eigen::VectorXd::Random(func->NumCoefficients());
    
    // evaluate the flux
    const Eigen::VectorXd flux = system->Flux(func, x, coeff);
    EXPECT_EQ(flux.size(), indim);
    const double eval = func->Evaluate(x, coeff) [0];
    const Eigen::VectorXd expectedFlux = 0.5*eval*eval*vel;
    EXPECT_NEAR((flux-expectedFlux).norm(), 0.0, 1.0e-14);

    // compute the divergence of the flux (the operator)
    const Eigen::VectorXd op = system->Operator(func, x, coeff);
  }
  
  /// The input dimension
  const std::size_t indim = 4;

  /// The velocity vector
  Eigen::VectorXd vel;

  /// Burgers' equation 
  std::shared_ptr<BurgersEquation> system;
};

TEST_F(BurgersEquationTests, ConstantVelocity) {
  vel = 2.5*Eigen::VectorXd::Ones(indim);
  system = std::make_shared<BurgersEquation>(indim, 2.5);
}

TEST_F(BurgersEquationTests, NonConstantVelocity) {
  vel = Eigen::VectorXd::Random(indim);
  system = std::make_shared<BurgersEquation>(vel);
}

  
  /*TEST(BurgersEquationTests, Flux) {
  const std::size_t indim = 4;

  BurgersEquation system(indim);
  EXPECT_EQ(system.indim, indim);
  EXPECT_EQ(system.outdim, 1);
  
  EXPECT_TRUE(false);
  }*/

} // namespace tests
} // namespace clf
