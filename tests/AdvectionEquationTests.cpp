#include <gtest/gtest.h>

#include "clf/Hypercube.hpp"
#include "clf/LegendrePolynomials.hpp"

#include "clf/AdvectionEquation.hpp"

namespace clf {
namespace tests {

/// A class to run the tests for clf::AdvectionEquation
class AdvectionEquationTests : public::testing::Test {
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
    const Eigen::VectorXd expectedFlux = vel*func->Evaluate(x, coeff) [0];
    EXPECT_NEAR((flux-expectedFlux).norm(), 0.0, 1.0e-14);

    // flux divergence
    const double div = system->FluxDivergence(func, x, coeff);
    const double divFD = system->FluxDivergenceFD(func, x, coeff);
    EXPECT_NEAR(div, divFD, 1.0e-12);

    std::cout << "div: " << div << " divFD: " << divFD << std::endl;
    

    // compute the divergence of the flux (the operator)
    const Eigen::VectorXd op = system->Operator(func, x, coeff);
  }

  /// The input dimension
  const std::size_t indim = 4;

  /// The velocity vector
  Eigen::VectorXd vel;

  /// The advection equation 
  std::shared_ptr<AdvectionEquation> system;
};

TEST_F(AdvectionEquationTests, ConstantVelocity) {
  vel = 2.5*Eigen::VectorXd::Ones(indim);
  system = std::make_shared<AdvectionEquation>(indim, 2.5);
}

TEST_F(AdvectionEquationTests, NonConstantVelocity) {
  vel = Eigen::VectorXd::Random(indim);
  system = std::make_shared<AdvectionEquation>(vel);
}

} // namespace tests
} // namespace clf
