#include <gtest/gtest.h>

#include "clf/LocalFunction.hpp"
#include "clf/LegendrePolynomials.hpp"
#include "clf/Hypercube.hpp"

using namespace clf;

TEST(LocalFunctionTests, Construction) {
  const std::size_t indim = 5;
  const std::size_t outdim = 3;
  const std::size_t maxOrder = 4;
  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);

  const double delta = 0.1;
  const Eigen::VectorXd xbar = Eigen::VectorXd::Random(indim);
  auto domain = std::make_shared<Hypercube>(xbar-Eigen::VectorXd::Constant(indim, delta), xbar+Eigen::VectorXd::Constant(indim, delta));
  
  auto basis = std::make_shared<LegendrePolynomials>();
  auto vec = std::make_shared<FeatureVector>(set, basis);
  auto mat = std::make_shared<FeatureMatrix>(vec, outdim, domain);

  LocalFunction func(mat);
  EXPECT_EQ(func.InputDimension(), indim);
  EXPECT_EQ(func.OutputDimension(), outdim);
  EXPECT_EQ(func.NumCoefficients(), mat->numBasisFunctions);

  const Eigen::VectorXd x = xbar + delta*Eigen::VectorXd::Random(indim);
  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(func.NumCoefficients());
  const Eigen::VectorXd eval = func.Evaluate(x, coeff);
  EXPECT_EQ(eval.size(), outdim);
  EXPECT_NEAR((eval-mat->ApplyTranspose(x, coeff)).norm(), 0.0, 1.0e-14);

  Eigen::MatrixXi counts = Eigen::MatrixXi::Zero(indim, 2);
  for( std::size_t i=0; i<counts.cols(); ++i ) {
    for( std::size_t j=0; j<3; ++j ) { ++counts(rand()%(indim-1), i); }
  }
  auto linOp = std::make_shared<LinearDifferentialOperator>(counts, outdim);
  const Eigen::MatrixXd diff = func.Derivative(x, coeff, linOp);
  EXPECT_EQ(diff.rows(), outdim);
  EXPECT_EQ(diff.cols(), counts.cols());
  EXPECT_NEAR((diff-mat->ApplyTranspose(x, coeff, linOp)).norm(), 0.0, 1.0e-14);

  for( std::size_t i=0; i<10; ++i ) { EXPECT_TRUE(domain->Inside(func.SampleDomain())); }
}
