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
