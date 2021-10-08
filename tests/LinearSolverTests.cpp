#include <gtest/gtest.h>

#include "clf/LinearSolver.hpp"

namespace clf {
namespace tests { 

/// A class to run the tests for clf::LinearSolver
class DenseLinearSolverTests : public::testing::Test {
protected:

  /// Set up the tests 
  virtual void SetUp() override {
    n = 25;
    m = 10;
    A = Eigen::MatrixXd::Random(n, m) + Eigen::MatrixXd::Identity(n, m);
  }

  /// The rows in the matrix
  std::size_t n;
  
  /// The columns in the matrix 
  std::size_t m;

  /// The matrix that defines the linear system 
  Eigen::MatrixXd A; 
};

TEST_F(DenseLinearSolverTests, Cholesky) {
  { // least squares solution
    auto linSolve = std::make_shared<DenseLinearSolver>(A, LinearSolverType::Cholesky, true);
    EXPECT_TRUE(linSolve->leastSq);
    EXPECT_EQ(linSolve->solver, LinearSolverType::Cholesky);

    const Eigen::VectorXd rhs = Eigen::VectorXd::Random(n);
    const Eigen::VectorXd soln = linSolve->Solve(rhs);
    EXPECT_NEAR((A.transpose()*(A*soln-rhs)).norm(), 0.0, 1.0e-13);
  }

  { // direct solve
    A = A.transpose()*A + Eigen::MatrixXd::Identity(m, m);
    auto linSolve = std::make_shared<DenseLinearSolver>(A, LinearSolverType::Cholesky, false);
    EXPECT_FALSE(linSolve->leastSq);
    EXPECT_EQ(linSolve->solver, LinearSolverType::Cholesky);

    const Eigen::VectorXd rhs = Eigen::VectorXd::Random(m);
    const Eigen::VectorXd soln = linSolve->Solve(rhs);
    EXPECT_NEAR((A*soln-rhs).norm(), 0.0, 1.0e-13);
  }
}

TEST_F(DenseLinearSolverTests, CholeskyPivot) {
  { // least squares solution
    auto linSolve = std::make_shared<DenseLinearSolver>(A, LinearSolverType::CholeskyPivot, true);
    EXPECT_TRUE(linSolve->leastSq);
    EXPECT_EQ(linSolve->solver, LinearSolverType::CholeskyPivot);

    const Eigen::VectorXd rhs = Eigen::VectorXd::Random(n);
    const Eigen::VectorXd soln = linSolve->Solve(rhs);
    EXPECT_NEAR((A.transpose()*(A*soln-rhs)).norm(), 0.0, 1.0e-14);
  }

  { // direct solve
    A = A.transpose()*A + Eigen::MatrixXd::Identity(m, m);
    auto linSolve = std::make_shared<DenseLinearSolver>(A, LinearSolverType::CholeskyPivot, false);
    EXPECT_FALSE(linSolve->leastSq);
    EXPECT_EQ(linSolve->solver, LinearSolverType::CholeskyPivot);

    const Eigen::VectorXd rhs = Eigen::VectorXd::Random(m);
    const Eigen::VectorXd soln = linSolve->Solve(rhs);
    EXPECT_NEAR((A*soln-rhs).norm(), 0.0, 1.0e-14);
  }
}

TEST_F(DenseLinearSolverTests, QR) {
  { // least squares solution
    auto linSolve = std::make_shared<DenseLinearSolver>(A, LinearSolverType::QR, true);
    EXPECT_TRUE(linSolve->leastSq);
    EXPECT_EQ(linSolve->solver, LinearSolverType::QR);

    const Eigen::VectorXd rhs = Eigen::VectorXd::Random(n);
    const Eigen::VectorXd soln = linSolve->Solve(rhs);
    EXPECT_NEAR((A.transpose()*(A*soln-rhs)).norm(), 0.0, 1.0e-13);
  }

  { // direct solve
    A = A.transpose()*A + Eigen::MatrixXd::Identity(m, m);
    auto linSolve = std::make_shared<DenseLinearSolver>(A, LinearSolverType::QR, false);
    EXPECT_FALSE(linSolve->leastSq);
    EXPECT_EQ(linSolve->solver, LinearSolverType::QR);

    const Eigen::VectorXd rhs = Eigen::VectorXd::Random(m);
    const Eigen::VectorXd soln = linSolve->Solve(rhs);
    EXPECT_NEAR((A*soln-rhs).norm(), 0.0, 1.0e-13);
  }
}

TEST_F(DenseLinearSolverTests, LU) {
  { // least squares solution
    auto linSolve = std::make_shared<DenseLinearSolver>(A, LinearSolverType::LU, true);
    EXPECT_TRUE(linSolve->leastSq);
    EXPECT_EQ(linSolve->solver, LinearSolverType::LU);

    const Eigen::VectorXd rhs = Eigen::VectorXd::Random(n);
    const Eigen::VectorXd soln = linSolve->Solve(rhs);
    EXPECT_NEAR((A.transpose()*(A*soln-rhs)).norm(), 0.0, 1.0e-13);
  }

  { // direct solve
    A = A.transpose()*A + Eigen::MatrixXd::Identity(m, m);
    auto linSolve = std::make_shared<DenseLinearSolver>(A, LinearSolverType::LU, false);
    EXPECT_FALSE(linSolve->leastSq);
    EXPECT_EQ(linSolve->solver, LinearSolverType::LU);

    const Eigen::VectorXd rhs = Eigen::VectorXd::Random(m);
    const Eigen::VectorXd soln = linSolve->Solve(rhs);
    EXPECT_NEAR((A*soln-rhs).norm(), 0.0, 1.0e-13);
  }
}

/// A class to run the tests for clf::LinearSolver
class SparseLinearSolverTests : public::testing::Test {
protected:

  /// Set up the tests 
  virtual void SetUp() override {
    n = 25;
    m = 10;
    A.resize(n, m);

    std::vector<Eigen::Triplet<double> > entries;
    for( std::size_t i=0; i<std::min(n, m); ++i ) { entries.emplace_back(i, i, 1.0); }
    for( std::size_t j=0; j<n*m/5; ++j ) { entries.emplace_back(rand()%n, rand()%m, 2.0*(double)rand()/RAND_MAX-1.0); }
    A.setFromTriplets(entries.begin(), entries.end());
  }

  /// The rows in the matrix
  std::size_t n;
  
  /// The columns in the matrix 
  std::size_t m;

  /// The matrix that defines the linear system 
  Eigen::SparseMatrix<double> A; 
};

TEST_F(SparseLinearSolverTests, Cholesky) {
  { // least squares solution
    auto linSolve = std::make_shared<SparseLinearSolver>(A, LinearSolverType::Cholesky, true);
    EXPECT_TRUE(linSolve->leastSq);
    EXPECT_EQ(linSolve->solver, LinearSolverType::Cholesky);

    const Eigen::VectorXd rhs = Eigen::VectorXd::Random(n);
    const Eigen::VectorXd soln = linSolve->Solve(rhs);
    EXPECT_NEAR((A.transpose()*(A*soln-rhs)).norm(), 0.0, 1.0e-13);
  }

  { // direct solve
    A = A.transpose()*A;
    for( std::size_t i=0; i<m; ++i ) { A.coeffRef(i, i) += 1.0; }
    auto linSolve = std::make_shared<SparseLinearSolver>(A, LinearSolverType::Cholesky, false);
    EXPECT_FALSE(linSolve->leastSq);
    EXPECT_EQ(linSolve->solver, LinearSolverType::Cholesky);

    const Eigen::VectorXd rhs = Eigen::VectorXd::Random(m);
    const Eigen::VectorXd soln = linSolve->Solve(rhs);
    EXPECT_NEAR((A*soln-rhs).norm(), 0.0, 1.0e-13);
  }
}

TEST_F(SparseLinearSolverTests, CholeskyPivot) {
  { // least squares solution
    auto linSolve = std::make_shared<SparseLinearSolver>(A, LinearSolverType::CholeskyPivot, true);
    EXPECT_TRUE(linSolve->leastSq);
    EXPECT_EQ(linSolve->solver, LinearSolverType::CholeskyPivot);

    const Eigen::VectorXd rhs = Eigen::VectorXd::Random(n);
    const Eigen::VectorXd soln = linSolve->Solve(rhs);
    EXPECT_NEAR((A.transpose()*(A*soln-rhs)).norm(), 0.0, 1.0e-14);
  }

  { // direct solve
    A = A.transpose()*A;
    for( std::size_t i=0; i<m; ++i ) { A.coeffRef(i, i) += 1.0; }
    auto linSolve = std::make_shared<SparseLinearSolver>(A, LinearSolverType::CholeskyPivot, false);
    EXPECT_FALSE(linSolve->leastSq);
    EXPECT_EQ(linSolve->solver, LinearSolverType::CholeskyPivot);

    const Eigen::VectorXd rhs = Eigen::VectorXd::Random(m);
    const Eigen::VectorXd soln = linSolve->Solve(rhs);
    EXPECT_NEAR((A*soln-rhs).norm(), 0.0, 1.0e-14);
  }
}

TEST_F(SparseLinearSolverTests, QR) {
  { // least squares solution
    auto linSolve = std::make_shared<SparseLinearSolver>(A, LinearSolverType::QR, true);
    EXPECT_TRUE(linSolve->leastSq);
    EXPECT_EQ(linSolve->solver, LinearSolverType::QR);

    const Eigen::VectorXd rhs = Eigen::VectorXd::Random(n);
    const Eigen::VectorXd soln = linSolve->Solve(rhs);
    EXPECT_NEAR((A.transpose()*(A*soln-rhs)).norm(), 0.0, 1.0e-13);
  }

  { // direct solve
    A = A.transpose()*A;
    for( std::size_t i=0; i<m; ++i ) { A.coeffRef(i, i) += 1.0; }
    A.makeCompressed();
    auto linSolve = std::make_shared<SparseLinearSolver>(A, LinearSolverType::QR, false);
    EXPECT_FALSE(linSolve->leastSq);
    EXPECT_EQ(linSolve->solver, LinearSolverType::QR);

    const Eigen::VectorXd rhs = Eigen::VectorXd::Random(m);
    const Eigen::VectorXd soln = linSolve->Solve(rhs);
    EXPECT_NEAR((A*soln-rhs).norm(), 0.0, 1.0e-13);
  }
}

TEST_F(SparseLinearSolverTests, LU) {
  { // least squares solution
    auto linSolve = std::make_shared<SparseLinearSolver>(A, LinearSolverType::LU, true);
    EXPECT_TRUE(linSolve->leastSq);
    EXPECT_EQ(linSolve->solver, LinearSolverType::LU);

    const Eigen::VectorXd rhs = Eigen::VectorXd::Random(n);
    const Eigen::VectorXd soln = linSolve->Solve(rhs);
    EXPECT_NEAR((A.transpose()*(A*soln-rhs)).norm(), 0.0, 1.0e-13);
  }

  { // direct solve
    A = A.transpose()*A;
    for( std::size_t i=0; i<m; ++i ) { A.coeffRef(i, i) += 1.0; }
    A.makeCompressed();
    auto linSolve = std::make_shared<SparseLinearSolver>(A, LinearSolverType::LU, false);
    EXPECT_FALSE(linSolve->leastSq);
    EXPECT_EQ(linSolve->solver, LinearSolverType::LU);

    const Eigen::VectorXd rhs = Eigen::VectorXd::Random(m);
    const Eigen::VectorXd soln = linSolve->Solve(rhs);
    EXPECT_NEAR((A*soln-rhs).norm(), 0.0, 1.0e-13);
  }
}

} // namespace tests 
} // namespace clf
