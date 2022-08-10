#include <gtest/gtest.h>

#include "clf/CLFExceptions.hpp"

#include "clf/Hypercube.hpp"

#include "clf/SystemOfEquations.hpp"

#include "clf/LegendrePolynomials.hpp"

using namespace clf;

TEST(SystemOfEquationsTests, DefaultImplementation) {
  const std::size_t indim = 2;
  const std::size_t outdim = 3;
  const std::size_t maxOrder = 4;
  
  SystemOfEquations sys(indim, outdim);

  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  const Eigen::VectorXd rhs = sys.RightHandSide(x);
  EXPECT_EQ(rhs.size(), outdim);
  EXPECT_DOUBLE_EQ(rhs.norm(), 0.0);

  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);

  const double delta = 0.1;
  const Eigen::VectorXd xbar = Eigen::VectorXd::Random(indim);
  auto domain = std::make_shared<Hypercube>(xbar-Eigen::VectorXd::Constant(indim, delta), xbar+Eigen::VectorXd::Constant(indim, delta));
  
  auto basis = std::make_shared<LegendrePolynomials>();
  auto vec = std::make_shared<FeatureVector>(set, basis);
  auto mat = std::make_shared<FeatureMatrix>(vec, outdim, domain);

  auto func = std::make_shared<LocalFunction>(mat);

  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(func->NumCoefficients());

  try { 
    sys.Operator(func, x, coeff);
  } catch( exceptions::NotImplemented const& exc ) {
    const std::string expected = "CLF Error: SystemOfEquations::Operator not yet implemented";
    const std::string err = exc.what();
    EXPECT_TRUE(err==expected);
  }
}
