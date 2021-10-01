#include <gtest/gtest.h>

#include "clf/Optimizer.hpp"
#include "clf/LevenbergMarquardt.hpp"
#include "clf/NLoptOptimizer.hpp"

namespace pt = boost::property_tree;
using namespace clf;

namespace clf { 
namespace tests {

/// An invalid clf::Optimizer that we implements pure virtual functions so that we can test other functionality
template<typename MatrixType>
class TestOptimizer : public Optimizer<MatrixType> {
public:

  /// Construct an optimizer with a <tt>nullptr</tt> cost function
  /**
  @param[in] pt Options for the algorithm
  */
  inline TestOptimizer(boost::property_tree::ptree const& pt) :
  Optimizer<MatrixType>(nullptr, pt)
  {}

  /// Implement the minimize function to do nothing so this class is not abstract
  inline virtual std::pair<Optimization::Convergence, double> Minimize(Eigen::VectorXd& beta) override { return std::pair<Optimization::Convergence, double>(Optimization::Convergence::FAILED, 0.0); }

  virtual ~TestOptimizer() = default;
private:
};

} // namespace tests
} // namespace clf 

TEST(OptimizerTests, ConstructLevenbergMarquardt_Dense) {
  pt::ptree options;
  options.put("Method", "LevenbergMarquardt");
  auto opt = Optimizer<Eigen::MatrixXd>::Construct(nullptr, options);
  EXPECT_TRUE(opt);

  auto lm = std::dynamic_pointer_cast<DenseLevenbergMarquardt>(opt);
  EXPECT_TRUE(lm);
}

TEST(OptimizerTests, ConstructNLopt_Dense) {
  pt::ptree options;
  options.put("Method", "NLopt");
  auto opt = Optimizer<Eigen::MatrixXd>::Construct(nullptr, options);
  EXPECT_TRUE(opt);

  auto nl = std::dynamic_pointer_cast<DenseNLoptOptimizer>(opt);
  EXPECT_TRUE(nl);
}

TEST(OptimizerTests, SolveLinearSystem_DenseQR) {
  // create an optimizer 
  pt::ptree pt;
  pt.put("LinearSolver", "QR");
  auto opt = std::make_shared<tests::TestOptimizer<Eigen::MatrixXd> >(pt);

  // the dimension and rank of the linear system 
  const std::size_t n = 100;
  const std::size_t rank = 85;

  { // square matrix (full rank)
    // create a linear system 
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(n, n) + Eigen::MatrixXd::Random(n, n);
    const Eigen::VectorXd b = Eigen::VectorXd::Random(n);
    const Eigen::VectorXd x = opt->SolveLinearSystem(A, b);
    EXPECT_EQ(x.size(), n);
    EXPECT_NEAR((A*x-b).norm(), 0.0, 1.0e-12);
  }

  { // square matrix (singular)
    // create a linear system 
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    A.block(0, 0, n, rank) = Eigen::MatrixXd::Identity(n, rank) + Eigen::MatrixXd::Random(n, rank);
    const Eigen::VectorXd b = Eigen::VectorXd::Random(n);
    const Eigen::VectorXd x = opt->SolveLinearSystem(A, b);
    EXPECT_EQ(x.size(), n);
    EXPECT_NEAR((A.transpose()*(A*x-b)).norm(), 0.0, 1.0e-12);
  }

  { // more cols than rows 
    // create a linear system 
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(rank, n) + Eigen::MatrixXd::Random(rank, n);
    const Eigen::VectorXd b = Eigen::VectorXd::Random(rank);
    const Eigen::VectorXd x = opt->SolveLinearSystem(A, b);
    EXPECT_EQ(x.size(), n);
    EXPECT_NEAR((A.transpose()*(A*x-b)).norm(), 0.0, 1.0e-11);
  }

  { // more rows than cols
    // create a linear system 
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(n, rank) + Eigen::MatrixXd::Random(n, rank);
    const Eigen::VectorXd b = Eigen::VectorXd::Random(n);
    const Eigen::VectorXd x = opt->SolveLinearSystem(A, b);
    EXPECT_EQ(x.size(), rank);
    EXPECT_NEAR((A.transpose()*(A*x-b)).norm(), 0.0, 1.0e-11);
  }
}

TEST(OptimizerTests, SolveLinearSystem_DenseLU) {
  // create an optimizer 
  pt::ptree pt;
  auto opt = std::make_shared<tests::TestOptimizer<Eigen::MatrixXd> >(pt);

  // the dimension of the linear system 
  std::size_t n = 100;

  // create a linear system 
  const Eigen::MatrixXd A = Eigen::MatrixXd::Identity(n, n) + Eigen::MatrixXd::Random(n, n);
  const Eigen::VectorXd b = Eigen::VectorXd::Random(n);
  const Eigen::VectorXd x = opt->SolveLinearSystem(A, b);
  EXPECT_EQ(x.size(), n);
  EXPECT_NEAR((A*x-b).norm(), 0.0, 1.0e-12);
}

TEST(OptimizerTests, FailedConstruction_Dense) {
  const std::string name = "BadOptimizerNameFails";

  pt::ptree options;
  options.put("Method", name);
  try {
    auto opt = Optimizer<Eigen::MatrixXd>::Construct(nullptr, options);
  } catch( exceptions::OptimizerNameException<Optimizer<Eigen::MatrixXd> > const& exc ) {
    EXPECT_EQ(exc.name, name);
  }
}

TEST(OptimizerTests, ConstructLevenbergMarquardt_Sparse) {
  pt::ptree options;
  options.put("Method", "LevenbergMarquardt");
  auto opt = Optimizer<Eigen::SparseMatrix<double> >::Construct(nullptr, options);
  EXPECT_TRUE(opt);

  auto lm = std::dynamic_pointer_cast<SparseLevenbergMarquardt>(opt);
  EXPECT_TRUE(lm);
}

TEST(OptimizerTests, ConstructNLopt_Sparse) {
  pt::ptree options;
  options.put("Method", "NLopt");
  auto opt = Optimizer<Eigen::SparseMatrix<double> >::Construct(nullptr, options);
  EXPECT_TRUE(opt);

  auto nl = std::dynamic_pointer_cast<SparseNLoptOptimizer>(opt);
  EXPECT_TRUE(nl);
}

TEST(OptimizerTests, SolveLinearSystem_Sparse) {
  // create an optimizer 
  pt::ptree pt;
  pt.put("LinearSolver", "QR");
  auto opt = std::make_shared<tests::TestOptimizer<Eigen::SparseMatrix<double> > >(pt);

  // the dimension and rank of the linear system 
  const std::size_t n = 100;
  const std::size_t rank = 85;

  { // square matrix (full rank)
    std::vector<Eigen::Triplet<double> > triplets;
    for( std::size_t i=0; i<n; ++i ) {
      triplets.emplace_back(i, i, 1.0);
      for( std::size_t j=0; j<n/10; ++j ) { triplets.emplace_back(i, rand()%n, (double)rand()/RAND_MAX); }
    }
    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    const Eigen::VectorXd b = Eigen::VectorXd::Random(n);
    const Eigen::VectorXd x = opt->SolveLinearSystem(A, b);
    EXPECT_EQ(x.size(), n);
    EXPECT_NEAR((A*x-b).norm(), 0.0, 1.0e-12);
  }

  { // square matrix (singular)
    std::vector<Eigen::Triplet<double> > triplets;
    for( std::size_t i=0; i<rank; ++i ) {
      triplets.emplace_back(i, i, 1.0);
      for( std::size_t j=0; j<n/10; ++j ) { triplets.emplace_back(i, rand()%n, (double)rand()/RAND_MAX); }
    }
    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    const Eigen::VectorXd b = Eigen::VectorXd::Random(n);
    const Eigen::VectorXd x = opt->SolveLinearSystem(A, b);
    EXPECT_EQ(x.size(), n);
    EXPECT_NEAR((A.transpose()*(A*x-b)).norm(), 0.0, 1.0e-12);
  }

  { // more cols than rows
    std::vector<Eigen::Triplet<double> > triplets;
    for( std::size_t i=0; i<rank; ++i ) {
      triplets.emplace_back(i, i, 1.0);
      for( std::size_t j=0; j<n/10; ++j ) { triplets.emplace_back(i, rand()%n, (double)rand()/RAND_MAX); }
    }
    Eigen::SparseMatrix<double> A(rank, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    const Eigen::VectorXd b = Eigen::VectorXd::Random(rank);
    const Eigen::VectorXd x = opt->SolveLinearSystem(A, b);
    EXPECT_EQ(x.size(), n);
    EXPECT_NEAR((A.transpose()*(A*x-b)).norm(), 0.0, 1.0e-12);
  }

  { // more rows than cols
    std::vector<Eigen::Triplet<double> > triplets;
    for( std::size_t i=0; i<n; ++i ) {
      if( i<rank ) { triplets.emplace_back(i, i, 1.0); }
      for( std::size_t j=0; j<n/10; ++j ) { triplets.emplace_back(i, rand()%rank, (double)rand()/RAND_MAX); }
    }
    Eigen::SparseMatrix<double> A(n, rank);
    A.setFromTriplets(triplets.begin(), triplets.end());
    const Eigen::VectorXd b = Eigen::VectorXd::Random(n);
    const Eigen::VectorXd x = opt->SolveLinearSystem(A, b);
    EXPECT_EQ(x.size(), rank);
    EXPECT_NEAR((A.transpose()*(A*x-b)).norm(), 0.0, 1.0e-12);
  }
}

TEST(OptimizerTests, SolveLinearSystem_SparseLU) {
  // create an optimizer 
  pt::ptree pt;
  auto opt = std::make_shared<tests::TestOptimizer<Eigen::SparseMatrix<double> > >(pt);

  // the dimension of the linear system 
  std::size_t n = 100;

  std::vector<Eigen::Triplet<double> > triplets;
  for( std::size_t i=0; i<n; ++i ) {
    triplets.emplace_back(i, i, 1.0);
    for( std::size_t j=0; j<n/10; ++j ) { triplets.emplace_back(i, rand()%n, (double)rand()/RAND_MAX); }
  }
  Eigen::SparseMatrix<double> A(n, n);
  A.setFromTriplets(triplets.begin(), triplets.end());
  const Eigen::VectorXd b = Eigen::VectorXd::Random(n);
  const Eigen::VectorXd x = opt->SolveLinearSystem(A, b);
  EXPECT_EQ(x.size(), n);
  EXPECT_NEAR((A*x-b).norm(), 0.0, 1.0e-12);
}

TEST(OptimizerTests, FailedConstruction_Sparse) {
  const std::string name = "BadOptimizerNameFails";

  pt::ptree options;
  options.put("Method", name);
  try {
    auto opt = Optimizer<Eigen::SparseMatrix<double> >::Construct(nullptr, options);
  } catch( exceptions::OptimizerNameException<Optimizer<Eigen::SparseMatrix<double> > > const& exc ) {
    EXPECT_EQ(exc.name, name);
  }
}
