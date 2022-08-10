#include <gtest/gtest.h>

#include "clf/LegendrePolynomials.hpp"
#include "clf/FeatureVector.hpp"
#include "clf/Hypercube.hpp"

using namespace clf;

TEST(FeatureVectorTests, EvaluateTest) {
  const std::size_t dim = 3;
  const std::size_t maxOrder = 4;
  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(dim, maxOrder);

  auto basis = std::make_shared<LegendrePolynomials>();

  const double delta = 0.5;
  const Eigen::VectorXd xbar = Eigen::VectorXd::Random(dim);
  auto domain = std::make_shared<Hypercube>(xbar-Eigen::VectorXd::Constant(dim, delta), xbar+Eigen::VectorXd::Constant(dim, delta));

  FeatureVector vec(set, basis, domain);
  EXPECT_EQ(vec.InputDimension(), set->Dimension());
  EXPECT_EQ(vec.NumBasisFunctions(), set->NumIndices());

  const Eigen::VectorXd x = Eigen::VectorXd::Random(dim);
  const Eigen::VectorXd y = domain->MapToHypercube(x);
  const Eigen::VectorXd eval = vec.Evaluate(x);

  Eigen::VectorXd expected = Eigen::VectorXd::Ones(set->NumIndices());
  for( std::size_t i=0; i<set->NumIndices(); ++i ) {
    for( std::size_t d=0; d<dim; ++d ) { expected(i) *= basis->Evaluate(set->indices[i].alpha[d], y(d)); }
  }

  EXPECT_EQ(eval.size(), expected.size());
  EXPECT_NEAR((eval-expected).norm(), 0.0, 1.0e-13);
}
