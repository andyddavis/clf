#include <gtest/gtest.h>

#include "clf/LegendrePolynomials.hpp"
#include "clf/Hypercube.hpp"
#include "clf/AdvectionEquation.hpp"
#include "clf/BurgersEquation.hpp"
#include "clf/ConservationLawWeakFormResidual.hpp"

namespace clf {
namespace tests {

/// A class to run the tests for clf::ConservationLawWeakFormResidual
class ConservationLawWeakFormResidualTests : public::testing::Test {
protected:
  /// Tear down the tests
  virtual void TearDown() override {
    const std::size_t numPoints = 100;
    
    auto para = std::make_shared<Parameters>();
    para->Add("NumPoints", numPoints);
    
    const double delta = 0.75;
    const Eigen::VectorXd xbar = Eigen::VectorXd::Random(indim);
    auto domain = std::make_shared<Hypercube>(xbar-Eigen::VectorXd::Constant(indim, delta), xbar+Eigen::VectorXd::Constant(indim, delta));
        
    // the local function that we are trying to fit
    const std::size_t maxOrder = 4;
    std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);
    auto basis = std::make_shared<LegendrePolynomials>();
    auto func = std::make_shared<LocalFunction>(set, basis, domain, 1);
    
    // the feature vector that defines the test function
    auto vec = std::make_shared<FeatureVector>(set, basis);
    
    auto resid = std::make_shared<ConservationLawWeakFormResidual>(func, system, vec, para);
    EXPECT_EQ(resid->indim, func->NumCoefficients());
    EXPECT_EQ(resid->outdim, func->NumCoefficients());
    EXPECT_EQ(resid->NumBoundaryPoints(), numPoints);
    EXPECT_EQ(resid->NumPoints(), numPoints);
    
    const Eigen::VectorXd coeff = Eigen::VectorXd::Random(resid->indim);
    
    Eigen::VectorXd expected = Eigen::VectorXd::Zero(resid->outdim);
    for( std::size_t i=0; i<resid->NumBoundaryPoints(); ++i ) {
      auto pt = resid->GetBoundaryPoint(i);
      expected += vec->Evaluate(pt->x)*pt->normal->dot(system->Flux(func, pt->x, coeff))/resid->NumBoundaryPoints();
    }
    for( std::size_t i=0; i<resid->NumPoints(); ++i ) {
      auto pt = resid->GetPoint(i);
      expected -= ( vec->Derivative(pt->x, Eigen::MatrixXi::Identity(indim, indim))*system->Flux(func, pt->x, coeff) + vec->Evaluate(pt->x)*system->RightHandSide(pt->x) [0] )/resid->NumPoints();
    }
    
    const Eigen::VectorXd eval = resid->Evaluate(coeff);
    EXPECT_EQ(eval.size(), resid->outdim);
    EXPECT_NEAR((eval-expected).norm(), 0.0, 1.0e-14);
    
    const Eigen::MatrixXd jac = resid->Jacobian(coeff);
    EXPECT_EQ(jac.rows(), vec->NumBasisFunctions());
    EXPECT_EQ(jac.cols(), func->NumCoefficients());
    const Eigen::MatrixXd jacFD = resid->JacobianFD(coeff);
    EXPECT_EQ(jacFD.rows(), vec->NumBasisFunctions());
    EXPECT_EQ(jacFD.cols(), func->NumCoefficients());
    EXPECT_NEAR((jac-jacFD).norm(), 0.0, 1.0e-10);
    
    const Eigen::VectorXd weights = Eigen::VectorXd::Random(resid->outdim);
    const Eigen::MatrixXd hess = resid->Hessian(coeff, weights);
    EXPECT_EQ(hess.rows(), func->NumCoefficients());
    EXPECT_EQ(hess.cols(), func->NumCoefficients());
    const Eigen::MatrixXd hessFD = resid->HessianFD(coeff, weights);
    EXPECT_EQ(hessFD.rows(), func->NumCoefficients());
    EXPECT_EQ(hessFD.cols(), func->NumCoefficients());
    EXPECT_NEAR((hess-hessFD).norm(), 0.0, 1.0e-10);
  }

  /// The spatial dimension
  const std::size_t indim = 3;

  /// The system of equations we are trying to satisfy
  std::shared_ptr<ConservationLaw> system;
};

TEST_F(ConservationLawWeakFormResidualTests, AdvectionEquation) {
  system = std::make_shared<AdvectionEquation>(indim, 2.5);
}

TEST_F(ConservationLawWeakFormResidualTests, BurgersEquation) {
  system = std::make_shared<BurgersEquation>(indim, 2.5);
}

} // namespace tests
} // namespace clf

