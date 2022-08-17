#include <gtest/gtest.h>

#include "clf/LegendrePolynomials.hpp"
#include "clf/FiniteDifference.hpp"

#include "clf/FeatureVector.hpp"

#include "TestDomains.hpp"

namespace clf {
namespace tests {

/// A class to run the tests for clf::FeatureVector
class FeatureVectorTests : public::testing::Test {
protected:

  /// Estimate the derivative of the feature vector using finite difference
  /**
     @param[in] vec The feature vector 
     @param[in] x The location where we are evalating the derivative 
     @param[in] counts The order of the derivatives
     \return The derivative of the feature vector using finite difference
   */
  Eigen::VectorXd DerivativeFD(std::shared_ptr<FeatureVector> const& vec, Eigen::VectorXd const& x, Eigen::VectorXi const& counts) {
    const double delta = 1.0e-3;
    const Eigen::VectorXd weights = FiniteDifference::Weights(8);
    Eigen::VectorXd xcopy = x; // the fd needs to modify this vector, so copy to remove const

    int ind;
    counts.maxCoeff(&ind);
    
    if( counts.sum()==1 ) {
      return FiniteDifference::Derivative<Eigen::VectorXd>(ind, delta, weights, xcopy, [&vec](Eigen::VectorXd const& x) { return vec->Evaluate(x); });
    }

    Eigen::VectorXi newCounts = counts;
    --newCounts(ind);
    
    return FiniteDifference::Derivative<Eigen::VectorXd>(ind, delta, weights, xcopy, [this, &vec, &newCounts](Eigen::VectorXd const& x) { return DerivativeFD(vec, x, newCounts); });
  }

  /// Set up the tests
  virtual void SetUp() override {
    set = MultiIndexSet::CreateTotalOrder(dim, maxOrder);
  }

  /// Tear down the tests
  virtual void TearDown() override {
    EXPECT_EQ(vec->InputDimension(), set->Dimension());
    EXPECT_EQ(vec->NumBasisFunctions(), set->NumIndices());
    
    const Eigen::VectorXd x = Eigen::VectorXd::Random(dim);
    const Eigen::VectorXd y = (domain? domain->MapToHypercube(x) : x);
    const Eigen::VectorXd eval = vec->Evaluate(x);

    Eigen::VectorXd expected = Eigen::VectorXd::Ones(set->NumIndices());
    for( std::size_t i=0; i<set->NumIndices(); ++i ) {
      for( std::size_t d=0; d<dim; ++d ) { expected(i) *= basis->Evaluate(set->indices[i].alpha[d], y(d)); }
    }

    EXPECT_EQ(eval.size(), expected.size());
    EXPECT_NEAR((eval-expected).norm(), 0.0, 1.0e-13);

    for( std::size_t i=0; i<dim; ++i ) {
      Eigen::VectorXi counts = Eigen::VectorXi::Zero(dim);
      counts(i) = 1;
      const Eigen::VectorXd deriv = vec->Derivative(x, counts);
      const Eigen::VectorXd derivFD = DerivativeFD(vec, x, counts);
      EXPECT_NEAR((deriv-derivFD).norm(), 0.0, 1.0e-10);
    }

    Eigen::VectorXi counts = Eigen::VectorXi::Zero(dim);
    for( std::size_t i=0; i<3; ++i ) { ++counts(rand()%(dim-1)); }
    const Eigen::VectorXd deriv = vec->Derivative(x, counts);
    const Eigen::VectorXd derivFD = DerivativeFD(vec, x, counts);
    EXPECT_NEAR((deriv-derivFD).norm()/deriv.norm(), 0.0, 1.0e-6);
  }

  /// The dimension of the space
  const std::size_t dim = 5;

  /// The maximum order of the basis
  const std::size_t maxOrder = 4;

  /// The multi-index set that defines the basis
  std::shared_ptr<MultiIndexSet> set;

  /// The component-wise basis functions 
  std::shared_ptr<LegendrePolynomials> basis = std::make_shared<LegendrePolynomials>();

  /// The domain
  std::shared_ptr<Domain> domain;

  /// The feature vector
  std::shared_ptr<FeatureVector> vec;

};

TEST_F(FeatureVectorTests, NoDomain) {
  vec = std::make_shared<FeatureVector>(set, basis);
}

TEST_F(FeatureVectorTests, HasDomain) {
  domain = std::make_shared<tests::TestHypercubeMapDomain>(dim);
  vec = std::make_shared<FeatureVector>(set, basis, domain);
}

} // namespace tests
} // namespace clf
