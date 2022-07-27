#include <gtest/gtest.h>

#include "clf/IdentityModel.hpp"
#include "clf/LinearModel.hpp"

#include "clf/LegendrePolynomials.hpp"
#include "clf/LocalResidual.hpp"

using namespace clf;

TEST(LocalResidualTests, IdentityModel) {
  const double radius = 0.1;
  const std::size_t numPoints = 100;
  const std::size_t indim = 3;
  const std::size_t outdim = 2;
  const Point point(Eigen::VectorXd::Random(indim));

  auto para = std::make_shared<Parameters>();
  para->Add<std::size_t>("NumPoints", numPoints);
  para->Add<double>("Radius", radius);

  const double delta = 0.75;
  const Eigen::VectorXd xbar = Eigen::VectorXd::Random(indim);

  const std::size_t maxOrder = 4;
  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);
  auto basis = std::make_shared<LegendrePolynomials>();
  auto func = std::make_shared<LocalFunction>(set, basis, delta, xbar, outdim);

  auto system = std::make_shared<IdentityModel>(indim, outdim);

  LocalResidual resid(func, system, point, para);
  EXPECT_EQ(resid.indim, func->NumCoefficients());
  EXPECT_EQ(resid.outdim, outdim*numPoints);
  EXPECT_EQ(resid.NumLocalPoints(), numPoints);

  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(resid.indim);

  const Eigen::VectorXd eval = resid.Evaluate(coeff);
  EXPECT_EQ(eval.size(), resid.outdim);
  std::size_t start = 0;
  for( std::size_t i=0; i<numPoints; ++i ) {
    EXPECT_NEAR((eval.segment(start, outdim)-func->Evaluate(resid.GetPoint(i), coeff)).norm(), 0.0, 1.0e-14);
    start += outdim;
  }

  const Eigen::MatrixXd jac = resid.Jacobian(coeff);
  EXPECT_EQ(jac.rows(), outdim*numPoints);
  EXPECT_EQ(jac.cols(), func->NumCoefficients());
  const Eigen::MatrixXd jacFD = resid.JacobianFD(coeff);
  EXPECT_EQ(jacFD.rows(), outdim*numPoints);
  EXPECT_EQ(jacFD.cols(), func->NumCoefficients());
  EXPECT_NEAR((jac-jacFD).norm(), 0.0, 1.0e-10);

  const Eigen::VectorXd weights = Eigen::VectorXd::Random(resid.outdim);
  const Eigen::MatrixXd hess = resid.Hessian(coeff, weights);
  EXPECT_EQ(hess.rows(), func->NumCoefficients());
  EXPECT_EQ(hess.cols(), func->NumCoefficients());
  const Eigen::MatrixXd hessFD = resid.HessianFD(coeff, weights);
  EXPECT_EQ(hessFD.rows(), func->NumCoefficients());
  EXPECT_EQ(hessFD.cols(), func->NumCoefficients());
  EXPECT_NEAR((hess-hessFD).norm(), 0.0, 1.0e-10);
}

TEST(LocalResidualTests, LinearModel) {
  const double radius = 0.1;
  const std::size_t numPoints = 100;
  const std::size_t indim = 3;
  const std::size_t outdim = 2;
  const Point point(Eigen::VectorXd::Random(indim));

  auto para = std::make_shared<Parameters>();
  para->Add<std::size_t>("NumPoints", numPoints);
  para->Add<double>("Radius", radius);

  const double delta = 0.75;
  const Eigen::VectorXd xbar = Eigen::VectorXd::Random(indim);

  const std::size_t maxOrder = 4;
  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);
  auto basis = std::make_shared<LegendrePolynomials>();
  auto func = std::make_shared<LocalFunction>(set, basis, delta, xbar, outdim);

  const Eigen::MatrixXd mat = Eigen::MatrixXd::Random(outdim, outdim);
  auto system = std::make_shared<LinearModel>(indim, mat);

  LocalResidual resid(func, system, point, para);
  EXPECT_EQ(resid.indim, func->NumCoefficients());
  EXPECT_EQ(resid.outdim, outdim*numPoints);
  EXPECT_EQ(resid.NumLocalPoints(), numPoints);

  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(resid.indim);

  const Eigen::VectorXd eval = resid.Evaluate(coeff);
  EXPECT_EQ(eval.size(), resid.outdim);
  std::size_t start = 0;
  for( std::size_t i=0; i<numPoints; ++i ) {
    EXPECT_NEAR((eval.segment(start, outdim)-mat*func->Evaluate(resid.GetPoint(i), coeff)).norm(), 0.0, 1.0e-14);
    start += outdim;
  }

  const Eigen::MatrixXd jac = resid.Jacobian(coeff);
  EXPECT_EQ(jac.rows(), outdim*numPoints);
  EXPECT_EQ(jac.cols(), func->NumCoefficients());
  const Eigen::MatrixXd jacFD = resid.JacobianFD(coeff);
  EXPECT_EQ(jacFD.rows(), outdim*numPoints);
  EXPECT_EQ(jacFD.cols(), func->NumCoefficients());
  EXPECT_NEAR((jac-jacFD).norm()/jac.norm(), 0.0, 1.0e-10);

  const Eigen::VectorXd weights = Eigen::VectorXd::Random(resid.outdim);
  const Eigen::MatrixXd hess = resid.Hessian(coeff, weights);
  EXPECT_EQ(hess.rows(), func->NumCoefficients());
  EXPECT_EQ(hess.cols(), func->NumCoefficients());
  const Eigen::MatrixXd hessFD = resid.HessianFD(coeff, weights);
  EXPECT_EQ(hessFD.rows(), func->NumCoefficients());
  EXPECT_EQ(hessFD.cols(), func->NumCoefficients());
  EXPECT_NEAR((hess-hessFD).norm(), 0.0, 1.0e-10);
}
