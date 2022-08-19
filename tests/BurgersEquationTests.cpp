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
    Eigen::VectorXd expectedFlux(indim);
    expectedFlux(0) = eval;
    expectedFlux.tail(indim-1) = 0.5*eval*eval*vel;
    EXPECT_NEAR((flux-expectedFlux).norm(), 0.0, 1.0e-14);

    // flux divergence
    const double div = system->FluxDivergence(func, x, coeff);
    const double divFD = system->FluxDivergenceFD(func, x, coeff);
    EXPECT_NEAR(div, divFD, 1.0e-12);

    const Eigen::VectorXd divGradWRTc = system->FluxDivergence_GradientWRTCoefficients(func, x, coeff);
    EXPECT_EQ(divGradWRTc.size(), coeff.size());
    const Eigen::VectorXd divGradWRTcFD = system->FluxDivergence_GradientWRTCoefficientsFD(func, x, coeff);
    EXPECT_EQ(divGradWRTcFD.size(), coeff.size());
    EXPECT_NEAR((divGradWRTc-divGradWRTcFD).norm(), 0.0, 1.0e-11);

    const Eigen::MatrixXd divHessWRTc = system->FluxDivergence_HessianWRTCoefficients(func, x, coeff);
    EXPECT_EQ(divHessWRTc.rows(), coeff.size());
    EXPECT_EQ(divHessWRTc.cols(), coeff.size());
    const Eigen::MatrixXd divHessWRTcFD = system->FluxDivergence_HessianWRTCoefficientsFD(func, x, coeff);
    EXPECT_EQ(divHessWRTcFD.rows(), coeff.size());
    EXPECT_EQ(divHessWRTcFD.cols(), coeff.size());
    EXPECT_NEAR((divHessWRTc-divHessWRTcFD).norm(), 0.0, 1.0e-10);

    // compute the divergence of the flux (the operator)
    const Eigen::VectorXd op = system->Operator(func, x, coeff);
    EXPECT_EQ(op.size(), 1);
    EXPECT_NEAR(std::abs(op(0)-div), 0.0, 1.0e-12);
    
    const Eigen::MatrixXd jac = system->JacobianWRTCoefficients(func, x, coeff);
    EXPECT_EQ(jac.rows(), 1);
    EXPECT_EQ(jac.cols(), coeff.size());
    const Eigen::MatrixXd jacFD = system->JacobianWRTCoefficientsFD(func, x, coeff);
    EXPECT_EQ(jacFD.rows(), 1);
    EXPECT_EQ(jacFD.cols(), coeff.size());
    EXPECT_NEAR((jac-jacFD).norm()/jac.norm(), 0.0, 1.0e-12);

    const Eigen::VectorXd weights = Eigen::VectorXd::Random(1);
    const Eigen::MatrixXd hess = system->HessianWRTCoefficients(func, x, coeff, weights);
    EXPECT_EQ(hess.rows(), coeff.size());
    EXPECT_EQ(hess.cols(), coeff.size());
    const Eigen::MatrixXd hessFD = system->HessianWRTCoefficientsFD(func, x, coeff, weights);
    EXPECT_EQ(hessFD.rows(), coeff.size());
    EXPECT_EQ(hessFD.cols(), coeff.size());
    EXPECT_NEAR((hess-hessFD).norm(), 0.0, 1.0e-10);
  }
  
  /// The input dimension
  const std::size_t indim = 4;

  /// The velocity vector
  Eigen::VectorXd vel;

  /// Burgers' equation 
  std::shared_ptr<BurgersEquation> system;
};

TEST_F(BurgersEquationTests, ConstantVelocity) {
  vel = 2.5*Eigen::VectorXd::Ones(indim-1);
  system = std::make_shared<BurgersEquation>(indim, 2.5);
}

TEST_F(BurgersEquationTests, NonConstantVelocity) {
  vel = Eigen::VectorXd::Random(indim-1);
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
