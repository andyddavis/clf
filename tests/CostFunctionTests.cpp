#include <gtest/gtest.h>

#include "clf/CostFunction.hpp"

#include "TestPenaltyFunctions.hpp"

using namespace clf;

TEST(CostFunctionTests, DenseMatrices) {
  // create the example penalty functions
  auto func0 = std::make_shared<tests::DensePenaltyFunctionTest0>();
  auto func1 = std::make_shared<tests::DensePenaltyFunctionTest1>();

  DenseCostFunction cost({func0, func1});
  EXPECT_EQ(cost.InputDimension(), func0->indim);
  EXPECT_EQ(cost.InputDimension(), func1->indim);
  EXPECT_EQ(cost.numPenaltyFunctions, 2);
  EXPECT_EQ(cost.numTerms, func0->outdim+func1->outdim);

  const Eigen::VectorXd beta = Eigen::VectorXd::Random(cost.InputDimension());
  const Eigen::VectorXd eval = cost.Evaluate(beta);
  EXPECT_EQ(eval.size(), cost.numTerms);

  Eigen::VectorXd expected(8);
  expected << beta(0), beta(0)*(1.0-beta(2)), 1.0-beta(1), 1.0-beta(1)+beta(2), beta(2), beta(2)*(1.0-beta(1)), beta(0)*beta(2), beta(0)*beta(0)*beta(1);
  EXPECT_NEAR((eval-expected).norm(), 0.0, 1.0e-14);

  const Eigen::MatrixXd jac = cost.Jacobian(beta);
  EXPECT_EQ(jac.cols(), func0->indim);
  EXPECT_EQ(jac.cols(), func1->indim);
  EXPECT_EQ(jac.rows(), func0->outdim+func1->outdim);
  EXPECT_NEAR((jac.topLeftCorner(func0->outdim, func0->indim)-func0->Jacobian(beta)).norm(), 0.0, 1.0e-14);
  EXPECT_NEAR((jac.bottomLeftCorner(func1->outdim, func1->indim)-func1->Jacobian(beta)).norm(), 0.0, 1.0e-14);

  const Eigen::VectorXd grad = cost.Gradient(eval, jac);
  EXPECT_EQ(grad.size(), func0->indim);  
  EXPECT_EQ(grad.size(), func1->indim);  
  EXPECT_NEAR((grad-cost.Gradient(beta)).norm(), 0.0, 1.0e-14);
  Eigen::VectorXd gradFD = Eigen::VectorXd::Zero(func0->indim);
  { // estimate the gradient with 8th order finite differences
    Eigen::VectorXd b = beta;
    Eigen::Vector4d weights(0.8, -0.2, 4.0/105.0, -1.0/280.0);  
    const double delta = 1.0e-2;
    for( std::size_t component=0; component<func0->indim; ++component ) {
      for( std::size_t j=0; j<weights.size(); ++j ) {
	b(component) += delta;
	const Eigen::VectorXd c = cost.Evaluate(b);
	gradFD(component) += weights(j)*c.dot(c);
      }
      b(component) -= weights.size()*delta;
      for( std::size_t j=0; j<weights.size(); ++j ) {
	b(component) -= delta;
	const Eigen::VectorXd c = cost.Evaluate(b);
	gradFD(component) -= weights(j)*c.dot(c);
      }
      b(component) += weights.size()*delta;
    }
    gradFD /= delta;
  }
  EXPECT_NEAR((grad-gradFD).norm(), 0.0, 1.0e-12);

  const Eigen::MatrixXd hess = cost.Hessian(beta);
  Eigen::MatrixXd hessFD = Eigen::MatrixXd::Zero(func0->indim, func0->indim);
  EXPECT_EQ(hess.rows(), hessFD.rows());
  EXPECT_EQ(hess.cols(), hessFD.cols());
  { // estimate the Hessian with 8th order finite differences
    Eigen::VectorXd b = beta;
    Eigen::Vector4d weights(0.8, -0.2, 4.0/105.0, -1.0/280.0);  
    const double delta = 1.0e-2;
    for( std::size_t component=0; component<func0->indim; ++component ) {
      for( std::size_t j=0; j<weights.size(); ++j ) {
	b(component) += delta;
	hessFD.col(component) += 2.0*weights(j)*cost.Jacobian(b).adjoint()*cost.Evaluate(b);
      }
      b(component) -= weights.size()*delta;
      for( std::size_t j=0; j<weights.size(); ++j ) {
	b(component) -= delta;
	hessFD.col(component) -= 2.0*weights(j)*cost.Jacobian(b).adjoint()*cost.Evaluate(b);
      }
      b(component) += weights.size()*delta;
    }
    hessFD /= delta;
  }
  EXPECT_NEAR((hess-hessFD).norm(), 0.0, 1.0e-12);
}

TEST(CostFunctionTests, SparseMatrices) {
  // create the example penalty functions
  auto func0 = std::make_shared<tests::SparsePenaltyFunctionTest0>();
  auto func1 = std::make_shared<tests::SparsePenaltyFunctionTest1>();

  SparseCostFunction cost({func0, func1});
  EXPECT_EQ(cost.InputDimension(), func0->indim);
  EXPECT_EQ(cost.InputDimension(), func1->indim);
  EXPECT_EQ(cost.numPenaltyFunctions, 2);
  EXPECT_EQ(cost.numTerms, func0->outdim+func1->outdim);

  const Eigen::VectorXd beta = Eigen::VectorXd::Random(cost.InputDimension());
  const Eigen::VectorXd eval = cost.Evaluate(beta);
  EXPECT_EQ(eval.size(), cost.numTerms);

  Eigen::VectorXd expected(8);
  expected << beta(0), beta(0)*(1.0-beta(2)), 1.0-beta(1), 1.0-beta(1)+beta(2), beta(2), beta(2)*(1.0-beta(1)), beta(0)*beta(2), beta(0)*beta(0)*beta(1);
  EXPECT_NEAR((eval-expected).norm(), 0.0, 1.0e-14);

  const Eigen::SparseMatrix<double> jac = cost.Jacobian(beta);
  EXPECT_EQ(jac.cols(), func0->indim);
  EXPECT_EQ(jac.cols(), func1->indim);
  EXPECT_EQ(jac.rows(), func0->outdim+func1->outdim);
  EXPECT_NEAR((jac.topLeftCorner(func0->outdim, func0->indim)-func0->Jacobian(beta)).norm(), 0.0, 1.0e-14);
  EXPECT_NEAR((jac.bottomLeftCorner(func1->outdim, func1->indim)-func1->Jacobian(beta)).norm(), 0.0, 1.0e-14);
}
