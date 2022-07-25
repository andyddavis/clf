#include <gtest/gtest.h>

#include "clf/SystemOfEquations.hpp"
#include "clf/LegendrePolynomials.hpp"

using namespace clf;

TEST(SystemOfEquationsTests, Construction) {
  const std::size_t indim = 4;
  const std::size_t outdim = 8;

  SystemOfEquations system(indim, outdim);
  EXPECT_EQ(system.indim, indim);
  EXPECT_EQ(system.outdim, outdim);

  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  const Eigen::VectorXd rhs = system.RightHandSide(x);
  EXPECT_EQ(rhs.size(), outdim);
  EXPECT_NEAR(rhs.norm(), 0.0, 1.0e-14);

  const std::size_t maxOrder = 4;
  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);
  
  auto basis = std::make_shared<LegendrePolynomials>();
  auto vec = std::make_shared<FeatureVector>(set, basis);
  auto mat = std::make_shared<FeatureMatrix>(vec, outdim);
  auto func = std::make_shared<LocalFunction>(mat);

  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(func->NumCoefficients());
  const Eigen::VectorXd eval = func->Evaluate(x, coeff);
  const Eigen::VectorXd op = system.Operator(func, x, coeff);
  EXPECT_EQ(op.size(), outdim);
  EXPECT_EQ(op.size(), eval.size());
  EXPECT_NEAR((eval-op).norm(), 0.0, 1.0e-14);
}
