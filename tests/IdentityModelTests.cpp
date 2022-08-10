#include <gtest/gtest.h>

#include "clf/Hypercube.hpp"
#include "clf/LegendrePolynomials.hpp"
#include "clf/IdentityModel.hpp"

using namespace clf;

TEST(IdentityModel, BasicTest) {
  const std::size_t indim = 4;
  const std::size_t outdim = 8;

  IdentityModel system(indim, outdim);
  EXPECT_EQ(system.indim, indim);
  EXPECT_EQ(system.outdim, outdim);

  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  const Eigen::VectorXd rhs = system.RightHandSide(x);
  EXPECT_EQ(rhs.size(), outdim);
  EXPECT_NEAR(rhs.norm(), 0.0, 1.0e-14);

  const std::size_t maxOrder = 4;
  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);

  const double delta = 0.5;
  const Eigen::VectorXd xbar = Eigen::VectorXd::Random(indim);
  auto domain = std::make_shared<Hypercube>(xbar-Eigen::VectorXd::Constant(indim, delta), xbar+Eigen::VectorXd::Constant(indim, delta));
  const Eigen::VectorXd y = domain->MapToHypercube(x);
  
  auto basis = std::make_shared<LegendrePolynomials>();
  auto vec = std::make_shared<FeatureVector>(set, basis);
  auto mat = std::make_shared<FeatureMatrix>(vec, outdim, domain);
  auto func = std::make_shared<LocalFunction>(mat);

  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(func->NumCoefficients());
  const Eigen::VectorXd eval = func->Evaluate(x, coeff);
  const Eigen::VectorXd op = system.Operator(func, x, coeff);
  EXPECT_EQ(op.size(), outdim);
  EXPECT_EQ(op.size(), eval.size());
  EXPECT_NEAR((eval-op).norm(), 0.0, 1.0e-14);

  const Eigen::MatrixXd jac = system.JacobianWRTCoefficients(func, x, coeff);
  EXPECT_EQ(jac.rows(), outdim);
  EXPECT_EQ(jac.cols(), func->NumCoefficients());
  const Eigen::MatrixXd jacFD = system.JacobianWRTCoefficientsFD(func, x, coeff);
  EXPECT_EQ(jacFD.rows(), outdim);
  EXPECT_EQ(jacFD.cols(), func->NumCoefficients());
  std::size_t start = 0;
  const Eigen::VectorXd phi = vec->Evaluate(y);
  for( std::size_t i=0; i<outdim; ++i ) {
    EXPECT_NEAR(jac.row(i).segment(0, start).norm(), 0.0, 1.0e-12);
    EXPECT_NEAR(jacFD.row(i).segment(0, start).norm(), 0.0, 1.0e-10);

    EXPECT_NEAR((jac.row(i).segment(start, phi.size()).transpose()-phi).norm(), 0.0, 1.0e-12);
    EXPECT_NEAR((jacFD.row(i).segment(start, phi.size()).transpose()-phi).norm(), 0.0, 1.0e-10);
    start += phi.size();

    EXPECT_NEAR(jac.row(i).segment(start, func->NumCoefficients()-start).norm(), 0.0, 1.0e-12);
    EXPECT_NEAR(jacFD.row(i).segment(start, func->NumCoefficients()-start).norm(), 0.0, 1.0e-10);
  }

  EXPECT_NEAR((jac-jacFD).norm()/jac.norm(), 0.0, 1.0e-10);

  const Eigen::VectorXd weights = Eigen::VectorXd::Random(outdim);
  const Eigen::MatrixXd hess = system.HessianWRTCoefficients(func, x, coeff, weights);
  EXPECT_EQ(hess.rows(), func->NumCoefficients());
  EXPECT_EQ(hess.cols(), func->NumCoefficients());
  EXPECT_NEAR(hess.norm(), 0.0, 1.0e-14);
  const Eigen::MatrixXd hessFD = system.HessianWRTCoefficientsFD(func, x, coeff, weights);
  EXPECT_EQ(hessFD.rows(), func->NumCoefficients());
  EXPECT_EQ(hessFD.cols(), func->NumCoefficients());
  EXPECT_NEAR(hessFD.norm(), 0.0, 1.0e-9);
}
