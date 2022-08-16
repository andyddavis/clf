#include <gtest/gtest.h>

#include "clf/LegendrePolynomials.hpp"
#include "clf/FiniteDifference.hpp"

#include "clf/FeatureVector.hpp"

#include "TestDomains.hpp"

using namespace clf;

Eigen::VectorXd DerivativeFD(FeatureVector const& vec, Eigen::VectorXd const& x, std::vector<std::size_t> const& B) {
  const double delta = 1.0e-3;
  const Eigen::VectorXd weights = FiniteDifference::Weights(8);
  Eigen::VectorXd xcopy = x; // the fd needs to modify this vector, so copy to remove const
      
  if( B.size()==1 ) {
    return FiniteDifference::Derivative<Eigen::VectorXd>(B[0], delta, weights, xcopy, [&vec](Eigen::VectorXd const& x) { return vec.Evaluate(x); });
  }
  
  return FiniteDifference::Derivative<Eigen::VectorXd>(B[0], delta, weights, xcopy, [&vec, &B](Eigen::VectorXd const& x) { return DerivativeFD(vec, x, std::vector<std::size_t>(B.begin()+1, B.end())); });
}

TEST(FeatureVectorTests, NoDomain) {
  const std::size_t dim = 6;
  const std::size_t maxOrder = 4;
  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(dim, maxOrder);

  auto basis = std::make_shared<LegendrePolynomials>();

  FeatureVector vec(set, basis);
  EXPECT_EQ(vec.InputDimension(), set->Dimension());
  EXPECT_EQ(vec.NumBasisFunctions(), set->NumIndices());

  const Eigen::VectorXd x = Eigen::VectorXd::Random(dim);
  const Eigen::VectorXd eval = vec.Evaluate(x);

  Eigen::VectorXd expected = Eigen::VectorXd::Ones(set->NumIndices());
  for( std::size_t i=0; i<set->NumIndices(); ++i ) {
    for( std::size_t d=0; d<dim; ++d ) { expected(i) *= basis->Evaluate(set->indices[i].alpha[d], x(d)); }
  }

  EXPECT_EQ(eval.size(), expected.size());
  EXPECT_NEAR((eval-expected).norm(), 0.0, 1.0e-10);

  for( std::size_t i=0; i<dim; ++i ) {
    const std::vector<std::size_t> B({i});
    const Eigen::VectorXd deriv = vec.Derivative(x, B);
    const Eigen::VectorXd derivFD = DerivativeFD(vec, x, B);
    EXPECT_NEAR((deriv-derivFD).norm(), 0.0, 1.0e-10);
  }
  
  const std::vector<std::size_t> B({rand()%(dim-1),
				    rand()%(dim-1),
				    rand()%(dim-1)});
  const Eigen::VectorXd deriv = vec.Derivative(x, B);
  const Eigen::VectorXd derivFD = DerivativeFD(vec, x, B);
  EXPECT_NEAR((deriv-derivFD).norm()/deriv.norm(), 0.0, 1.0e-7);
}

TEST(FeatureVectorTests, Domain) {
  const std::size_t dim = 2;
  const std::size_t maxOrder = 4;
  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(dim, maxOrder);

  auto basis = std::make_shared<LegendrePolynomials>();
  auto domain = std::make_shared<tests::TestHypercubeMapDomain>(dim);

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

  for( std::size_t i=0; i<dim; ++i ) {
    const std::vector<std::size_t> B({i});
    const Eigen::VectorXd deriv = vec.Derivative(x, B);
    const Eigen::VectorXd derivFD = DerivativeFD(vec, x, B);
    EXPECT_NEAR((deriv-derivFD).norm(), 0.0, 1.0e-10);
  }
  
  const std::vector<std::size_t> B({rand()%(dim-1),
				    rand()%(dim-1),
				    rand()%(dim-1)});
  const Eigen::VectorXd deriv = vec.Derivative(x, B);
  const Eigen::VectorXd derivFD = DerivativeFD(vec, x, B);
  EXPECT_NEAR((deriv-derivFD).norm()/deriv.norm(), 0.0, 1.0e-7);
}
