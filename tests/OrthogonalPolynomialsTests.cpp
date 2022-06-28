#include <gtest/gtest.h>

#include <cmath>

#include "clf/LegendrePolynomials.hpp"

using namespace clf;

TEST(OrthogonalPolynomialsTests, LegendrePolynomials) {
  LegendrePolynomials len;

  // a random number in [-1, 1]
  double x = 2.0*rand()/RAND_MAX + 1.0;

  // the constant polynomial is phi(x)=1
  EXPECT_DOUBLE_EQ(len.Evaluate(0, x), 1.0);

  // the linear polynomial is phi(x)=x
  EXPECT_DOUBLE_EQ(len.Evaluate(1, x), x);

  // try a known x for higher order
  EXPECT_DOUBLE_EQ(0.3, len.Evaluate(1, 0.3));
  EXPECT_DOUBLE_EQ(0.5*(3.0*std::pow(0.3, 2.0)-1.0), len.Evaluate(2, 0.3));

  const Eigen::VectorXd eval = len.EvaluateAll(5, 0.325);
  EXPECT_EQ(eval.size(), 6);
  EXPECT_DOUBLE_EQ(0.3375579333496094, len.Evaluate(5, 0.325));
  EXPECT_DOUBLE_EQ(1.0, eval(0));
  EXPECT_DOUBLE_EQ(0.325, eval(1));
  EXPECT_DOUBLE_EQ(0.3375579333496094, eval(5));

  EXPECT_NEAR(-0.05346106275520913, len.Evaluate(20, -0.845), 1.0e-14);
  EXPECT_NEAR(-0.1119514835092105, len.Evaluate(50, 0.1264), 1e-14);
  EXPECT_NEAR(-0.001892916076323403, len.Evaluate(200, -0.3598), 1e-14);
  EXPECT_NEAR(0.01954143166718206, len.Evaluate(1000, 0.4587), 1e-14);

  // evaluate derivatives at a known location
  x = 0.23;
  const Eigen::MatrixXd derivs = len.EvaluateAllDerivatives(4, x, 3);

  // first derivatives at a known location
  EXPECT_DOUBLE_EQ(0.0, len.EvaluateDerivative(0, x, 1));
  EXPECT_DOUBLE_EQ(0.0, derivs(0, 0));
  EXPECT_DOUBLE_EQ(1.0, len.EvaluateDerivative(1, x, 1));
  EXPECT_DOUBLE_EQ(1.0, derivs(1, 0));
  EXPECT_DOUBLE_EQ(3.0*x, len.EvaluateDerivative(2, x, 1));
  EXPECT_DOUBLE_EQ(3.0*x, derivs(2, 0));
  EXPECT_DOUBLE_EQ(7.5*x*x - 1.5, len.EvaluateDerivative(3, x, 1));
  EXPECT_DOUBLE_EQ(7.5*x*x - 1.5, derivs(3, 0));
  EXPECT_DOUBLE_EQ(17.5*std::pow(x, 3.0) - 7.5*x, len.EvaluateDerivative(4, x, 1));
  EXPECT_DOUBLE_EQ(17.5*std::pow(x, 3.0) - 7.5*x, derivs(4, 0));

  // second derivatives
  EXPECT_DOUBLE_EQ(0.0, len.EvaluateDerivative(0, x, 2));
  EXPECT_DOUBLE_EQ(0.0, derivs(0, 1));
  EXPECT_DOUBLE_EQ(0.0, len.EvaluateDerivative(1, x, 2));
  EXPECT_DOUBLE_EQ(0.0, derivs(1, 1));
  EXPECT_DOUBLE_EQ(3.0, len.EvaluateDerivative(2, x, 2));
  EXPECT_DOUBLE_EQ(3.0, derivs(2, 1));
  EXPECT_DOUBLE_EQ(15.0*x, len.EvaluateDerivative(3, x, 2));
  EXPECT_DOUBLE_EQ(15.0*x, derivs(3, 1));
  EXPECT_DOUBLE_EQ(52.5*std::pow(x, 2.0) - 7.5, len.EvaluateDerivative(4, x, 2));
  EXPECT_DOUBLE_EQ(52.5*std::pow(x, 2.0) - 7.5, derivs(4, 1));

  // third derivatives
  EXPECT_DOUBLE_EQ(0.0, len.EvaluateDerivative(0, x, 3));
  EXPECT_DOUBLE_EQ(0.0, derivs(0, 2));
  EXPECT_DOUBLE_EQ(0.0, len.EvaluateDerivative(1, x, 3));
  EXPECT_DOUBLE_EQ(0.0, derivs(1, 2));
  EXPECT_DOUBLE_EQ(0.0, len.EvaluateDerivative(2, x, 3));
  EXPECT_DOUBLE_EQ(0.0, derivs(2, 2));
  EXPECT_DOUBLE_EQ(15.0, len.EvaluateDerivative(3, x, 3));
  EXPECT_DOUBLE_EQ(15.0, derivs(3, 2));
  EXPECT_DOUBLE_EQ(105.0*x, len.EvaluateDerivative(4, x, 3));
  EXPECT_DOUBLE_EQ(105.0*x, derivs(4, 2));
}
