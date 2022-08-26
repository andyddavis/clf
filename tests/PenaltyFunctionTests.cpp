#include <gtest/gtest.h>

#include "TestPenaltyFunctions.hpp"

using namespace clf;

TEST(PenaltyFunctionTests, DenseTest0) {
  // create the example penalty function
  tests::DensePenaltyFunctionTest0 func;

  // the input and output dimensions are hardcoded as 3 and 2, respectively
  EXPECT_EQ(func.InputDimension(), 3);
  EXPECT_EQ(func.OutputDimension(), 2);

  // evaluate the cost function 
  const Eigen::VectorXd beta = Eigen::VectorXd::Random(3);
  const Eigen::VectorXd eval = func.Evaluate(beta);
  EXPECT_EQ(eval.size(), func.OutputDimension());
  EXPECT_NEAR(eval(0), beta(0), 1.0e-14);
  EXPECT_NEAR(eval(1), beta(0)*(1.0-beta(2)), 1.0e-14);

  const Eigen::MatrixXd exactJac = func.Jacobian(beta);
  EXPECT_EQ(exactJac.rows(), 2);
  EXPECT_EQ(exactJac.cols(), 3);  
  const Eigen::MatrixXd fdJac = func.JacobianFD(beta);
  EXPECT_EQ(exactJac.rows(), fdJac.rows());
  EXPECT_EQ(exactJac.cols(), fdJac.cols());
  EXPECT_NEAR((exactJac-fdJac).norm(), 0.0, 1.0e-13);

  const Eigen::MatrixXd weights = Eigen::VectorXd::Random(2);
  const Eigen::MatrixXd exactHess = func.Hessian(beta, weights);
  EXPECT_EQ(exactHess.rows(), 3);
  EXPECT_EQ(exactHess.cols(), 3);
  const Eigen::MatrixXd fdHess = func.HessianFD(beta, weights);
  EXPECT_EQ(exactHess.rows(), fdHess.rows());
  EXPECT_EQ(exactHess.cols(), fdHess.cols());
  EXPECT_NEAR((exactHess-fdHess).norm(), 0.0, 1.0e-13);
}

TEST(PenaltyFunctionTests, DenseTest1) {
  // create the example penalty function
  tests::DensePenaltyFunctionTest1 func;

  // the input and output dimensions are hardcoded as 3 and 2, respectively
  EXPECT_EQ(func.InputDimension(), 3);
  EXPECT_EQ(func.OutputDimension(), 6);

  // evaluate the cost function 
  const Eigen::VectorXd beta = Eigen::VectorXd::Random(3);
  const Eigen::VectorXd eval = func.Evaluate(beta);
  EXPECT_EQ(eval.size(), func.OutputDimension());
  EXPECT_NEAR(eval(0), 1.0-beta(1), 1.0e-14);
  EXPECT_NEAR(eval(1), 1.0-beta(1)+beta(2), 1.0e-14);
  EXPECT_NEAR(eval(2), beta(2), 1.0e-14);
  EXPECT_NEAR(eval(3), beta(2)*(1.0-beta(1)), 1.0e-14);
  EXPECT_NEAR(eval(4), beta(0)*beta(2), 1.0e-14);
  EXPECT_NEAR(eval(5), beta(0)*beta(0)*beta(1), 1.0e-13);

  const Eigen::MatrixXd exactJac = func.Jacobian(beta);
  EXPECT_EQ(exactJac.rows(), 6);
  EXPECT_EQ(exactJac.cols(), 3);  
  const Eigen::MatrixXd fdJac = func.JacobianFD(beta);
  EXPECT_EQ(exactJac.rows(), fdJac.rows());
  EXPECT_EQ(exactJac.cols(), fdJac.cols());
  EXPECT_NEAR((exactJac-fdJac).norm(), 0.0, 1.0e-13);

  const Eigen::MatrixXd weights = Eigen::VectorXd::Random(6);
  const Eigen::MatrixXd exactHess = func.Hessian(beta, weights);
  EXPECT_EQ(exactHess.rows(), 3);
  EXPECT_EQ(exactHess.cols(), 3);
  const Eigen::MatrixXd fdHess = func.HessianFD(beta, weights);
  EXPECT_EQ(exactHess.rows(), fdHess.rows());
  EXPECT_EQ(exactHess.cols(), fdHess.cols());
  EXPECT_NEAR((exactHess-fdHess).norm(), 0.0, 1.0e-13);
}

TEST(PenaltyFunctionTests, SparseTest0) {
  // create the example penalty function
  tests::SparsePenaltyFunctionTest0 func;

  // the input and output dimensions are hardcoded as 3 and 2, respectively
  EXPECT_EQ(func.InputDimension(), 3);
  EXPECT_EQ(func.OutputDimension(), 2);

  // evaluate the cost function 
  const Eigen::VectorXd beta = Eigen::VectorXd::Random(3);
  const Eigen::VectorXd eval = func.Evaluate(beta);
  EXPECT_EQ(eval.size(), func.OutputDimension());
  EXPECT_NEAR(eval(0), beta(0), 1.0e-14);
  EXPECT_NEAR(eval(1), beta(0)*(1.0-beta(2)), 1.0e-14);

  const Eigen::SparseMatrix<double> exactJac = func.Jacobian(beta);
  EXPECT_EQ(exactJac.rows(), 2);
  EXPECT_EQ(exactJac.cols(), 3);  
  const Eigen::SparseMatrix<double> fdJac = func.JacobianFD(beta);
  EXPECT_EQ(exactJac.rows(), fdJac.rows());
  EXPECT_EQ(exactJac.cols(), fdJac.cols());
  EXPECT_NEAR((exactJac-fdJac).norm(), 0.0, 1.0e-13);

  const Eigen::VectorXd weights = Eigen::VectorXd::Random(2);
  const Eigen::SparseMatrix<double> exactHess = func.Hessian(beta, weights);
  EXPECT_EQ(exactHess.rows(), 3);
  EXPECT_EQ(exactHess.cols(), 3);
  const Eigen::SparseMatrix<double> fdHess = func.HessianFD(beta, weights);
  EXPECT_EQ(exactHess.rows(), fdHess.rows());
  EXPECT_EQ(exactHess.cols(), fdHess.cols());
  EXPECT_NEAR((exactHess-fdHess).norm(), 0.0, 1.0e-13);
}

TEST(PenaltyFunctionTests, SparseTest1) {
  // create the example penalty function
  tests::SparsePenaltyFunctionTest1 func;

  // the input and output dimensions are hardcoded as 3 and 2, respectively
  EXPECT_EQ(func.InputDimension(), 3);
  EXPECT_EQ(func.OutputDimension(), 6);

  // evaluate the cost function 
  const Eigen::VectorXd beta = Eigen::VectorXd::Random(3);
  const Eigen::VectorXd eval = func.Evaluate(beta);
  EXPECT_EQ(eval.size(), func.OutputDimension());
  EXPECT_NEAR(eval(0), 1.0-beta(1), 1.0e-14);
  EXPECT_NEAR(eval(1), 1.0-beta(1)+beta(2), 1.0e-14);
  EXPECT_NEAR(eval(2), beta(2), 1.0e-14);
  EXPECT_NEAR(eval(3), beta(2)*(1.0-beta(1)), 1.0e-14);
  EXPECT_NEAR(eval(4), beta(0)*beta(2), 1.0e-14);
  EXPECT_NEAR(eval(5), beta(0)*beta(0)*beta(1), 1.0e-14);

  const Eigen::SparseMatrix<double> exactJac = func.Jacobian(beta);
  EXPECT_EQ(exactJac.rows(), 6);
  EXPECT_EQ(exactJac.cols(), 3);  
  const Eigen::SparseMatrix<double> fdJac = func.JacobianFD(beta);
  EXPECT_EQ(exactJac.rows(), fdJac.rows());
  EXPECT_EQ(exactJac.cols(), fdJac.cols());
  EXPECT_NEAR((exactJac-fdJac).norm(), 0.0, 1.0e-13);

  const Eigen::VectorXd weights = Eigen::VectorXd::Random(6);
  const Eigen::SparseMatrix<double> exactHess = func.Hessian(beta, weights);
  EXPECT_EQ(exactHess.rows(), 3);
  EXPECT_EQ(exactHess.cols(), 3);
  const Eigen::SparseMatrix<double> fdHess = func.HessianFD(beta, weights);
  EXPECT_EQ(exactHess.rows(), fdHess.rows());
  EXPECT_EQ(exactHess.cols(), fdHess.cols());
  EXPECT_NEAR((exactHess-fdHess).norm(), 0.0, 1.0e-12);
}
