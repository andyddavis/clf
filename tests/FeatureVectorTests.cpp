#include <gtest/gtest.h>

#include "clf/LegendrePolynomials.hpp"
#include "clf/FeatureVector.hpp"

using namespace clf;

TEST(FeatureVectorTests, EvaluateTest) {
  const std::size_t dim = 3;
  const std::size_t maxOrder = 4;
  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(dim, maxOrder);

  auto basis = std::make_shared<LegendrePolynomials>();

  FeatureVector vec(set, basis);
  EXPECT_EQ(vec.NumBasisFunctions(), set->NumIndices());

  const Eigen::VectorXd x = Eigen::VectorXd::Random(dim);
  const Eigen::VectorXd eval = vec.Evaluate(x);

  Eigen::VectorXd expected = Eigen::VectorXd::Ones(set->NumIndices());
  for( std::size_t i=0; i<set->NumIndices(); ++i ) {
    for( std::size_t d=0; d<dim; ++d ) { expected(i) *= basis->Evaluate(set->indices[i]->alpha[d], x(d)); }
  }

  EXPECT_EQ(eval.size(), expected.size());
  EXPECT_NEAR((eval-expected).norm(), 0.0, 1.0e-14);
}