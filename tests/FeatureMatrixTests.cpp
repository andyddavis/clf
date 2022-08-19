#include <gtest/gtest.h>

#include "clf/LegendrePolynomials.hpp"
#include "clf/FeatureMatrix.hpp"
#include "clf/Hypercube.hpp"

namespace clf {
namespace tests {

/// A class to run the tests for clf::FeatureMatrix
class FeatureMatrixTests : public::testing::Test {
protected:
  /// Set up the tests
  virtual inline void SetUp() override {
    basis = std::make_shared<LegendrePolynomials>();

    const double delta = 0.6;
    const Eigen::VectorXd xbar = Eigen::VectorXd::Random(indim);
    domain = std::make_shared<Hypercube>(xbar-Eigen::VectorXd::Constant(indim, delta), xbar+Eigen::VectorXd::Constant(indim, delta));
  }

  /// Test applying the feature matrix with a linear differential operator
  /**
     @param[in] vec The feature vectors that define the feature matrix
   */
  inline void TestDerivatives(std::vector<std::shared_ptr<const FeatureVector> > const& vecs) {
    {
      Eigen::MatrixXi count = Eigen::MatrixXi::Zero(indim, 2);
      for( std::size_t i=0; i<count.cols(); ++i ) {
	for( std::size_t j=0; j<2; ++j ) { ++count(rand()%indim, i); }
      }
      auto linOper = std::make_shared<LinearDifferentialOperator>(count, outdim);
      TestDerivatives(linOper, vecs);
    }
    
    {
      std::vector<Eigen::MatrixXi> counts(outdim);
      for( std::size_t i=0; i<outdim; ++i ) {
	counts[i] = Eigen::VectorXi::Zero(indim);
	for( std::size_t j=0; j<2; ++j ) { ++counts[i](rand()%(indim-1)); }
      }
      auto linOper = std::make_shared<LinearDifferentialOperator>(counts);
      TestDerivatives(linOper, vecs);
    }

    {
      std::vector<LinearDifferentialOperator::CountPair> counts(2);
      counts[0] = LinearDifferentialOperator::CountPair(Eigen::VectorXi::Zero(indim), outdim-1);
      counts[1] = LinearDifferentialOperator::CountPair(Eigen::VectorXi::Zero(indim), 1);
      auto linOper = std::make_shared<LinearDifferentialOperator>(counts);
      TestDerivatives(linOper, vecs);
    }
  }

  /// Test applying the feature matrix with a linear differential operator
  /**
     @param[in] linOper The linear differential operator
     @param[in] vec The feature vectors that define the feature matrix
   */
  inline void TestDerivatives(std::shared_ptr<LinearDifferentialOperator> const& linOper, std::vector<std::shared_ptr<const FeatureVector> > const& vecs) {
    const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);
    const Eigen::VectorXd y = domain->MapToHypercube(x);
    const Eigen::VectorXd coeff = Eigen::VectorXd::Random(mat->numBasisFunctions);

    const Eigen::VectorXd jac = domain->MapToHypercubeJacobian();

    const Eigen::MatrixXd output = mat->ApplyTranspose(x, coeff, linOper);

    EXPECT_EQ(output.rows(), outdim);
    EXPECT_EQ(output.cols(), linOper->NumOperators());
    Eigen::MatrixXd expected(outdim, linOper->NumOperators());
    std::size_t start = 0;
    for( std::size_t i=0; i<outdim; ++i ) {
      for( std::size_t j=0; j<linOper->NumOperators(); ++j ) {
	const Eigen::VectorXd eval = vecs[i]->Derivative(y, linOper->Counts(i).first.col(j), jac);
	expected(i, j) = eval.dot(coeff.segment(start, vecs[i]->NumBasisFunctions()));
      }
      start += vecs[i]->NumBasisFunctions();
    }
    EXPECT_NEAR((output-expected).norm()/output.norm(), 0.0, 1.0e-12);
  }
  
  /// The input dimension
  const std::size_t indim = 5;

  /// The output dimension
  std::size_t outdim = 3;

  /// The basis vectors
  std::shared_ptr<LegendrePolynomials> basis;

  /// The domain
  std::shared_ptr<Domain> domain;

  /// The feature matrix
  std::shared_ptr<FeatureMatrix> mat;
};

TEST_F(FeatureMatrixTests, SingleFeatureVectorTest) {
  const std::size_t maxOrder = 4;
  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);
  
  auto vec = std::make_shared<FeatureVector>(set, basis);

  mat = std::make_shared<FeatureMatrix>(vec, outdim, domain);
  EXPECT_EQ(mat->numBasisFunctions, outdim*set->NumIndices());
  EXPECT_EQ(mat->InputDimension(), indim);
  EXPECT_EQ(mat->numFeatureVectors, outdim);

  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);
  const Eigen::VectorXd y = domain->MapToHypercube(x);
  const Eigen::VectorXd eval = vec->Evaluate(y);

  for( std::size_t i=0; i<outdim; ++i ) { EXPECT_NEAR((mat->GetFeatureVector(i)->Evaluate(y)-eval).norm(), 0.0, 1.0e-14); }

  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(mat->numBasisFunctions);
  const Eigen::VectorXd output = mat->ApplyTranspose(x, coeff);
  EXPECT_EQ(output.size(), outdim);

  Eigen::VectorXd expected(outdim);
  for( std::size_t i=0; i<outdim; ++i ) { expected(i) = eval.dot(coeff.segment(i*set->NumIndices(), set->NumIndices())); }
  EXPECT_NEAR((output-expected).norm(), 0.0, 1.0e-14);

  TestDerivatives(std::vector<std::shared_ptr<const FeatureVector> >(outdim, vec));
}

TEST_F(FeatureMatrixTests, MultiFeatureVectorTest) {
  outdim = 2; // reset the output dimension for this test

  const std::size_t maxOrder1 = 4;
  std::shared_ptr<MultiIndexSet> set1 = MultiIndexSet::CreateTotalOrder(indim, maxOrder1);
  const std::size_t maxOrder2 = 8;
  std::shared_ptr<MultiIndexSet> set2 = MultiIndexSet::CreateTotalOrder(indim, maxOrder2);
  
  auto vec1 = std::make_shared<FeatureVector>(set1, basis);
  auto vec2 = std::make_shared<FeatureVector>(set2, basis);

  mat = std::make_shared<FeatureMatrix>(std::vector<std::shared_ptr<const FeatureVector> >{vec1, vec2}, domain);
  EXPECT_EQ(mat->numBasisFunctions, set1->NumIndices() + set2->NumIndices());
  EXPECT_EQ(mat->InputDimension(), indim);
  EXPECT_EQ(mat->numFeatureVectors, outdim);

  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);
  const Eigen::VectorXd y = domain->MapToHypercube(x);

  EXPECT_NEAR((mat->GetFeatureVector(0)->Evaluate(y)-vec1->Evaluate(y)).norm(), 0.0, 1.0e-14);
  EXPECT_NEAR((mat->GetFeatureVector(1)->Evaluate(y)-vec2->Evaluate(y)).norm(), 0.0, 1.0e-14);

  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(mat->numBasisFunctions);
  const Eigen::VectorXd output = mat->ApplyTranspose(x, coeff);
  EXPECT_EQ(output.size(), outdim);

  Eigen::VectorXd expected(outdim);
  expected(0) = vec1->Evaluate(y).dot(coeff.head(set1->NumIndices()));
  expected(1) = vec2->Evaluate(y).dot(coeff.tail(set2->NumIndices()));
  EXPECT_NEAR((output-expected).norm(), 0.0, 1.0e-14);

  TestDerivatives({vec1, vec2});
}

TEST_F(FeatureMatrixTests, RepeatedFeatureVectorsTest) {
  const std::size_t maxOrder1 = 4;
  std::shared_ptr<MultiIndexSet> set1 = MultiIndexSet::CreateTotalOrder(indim, maxOrder1);
  const std::size_t maxOrder2 = 8;
  std::shared_ptr<MultiIndexSet> set2 = MultiIndexSet::CreateTotalOrder(indim, maxOrder2);
    
  const FeatureMatrix::VectorPair vec1 = FeatureMatrix::VectorPair(std::make_shared<FeatureVector>(set1, basis), 1);
  const FeatureMatrix::VectorPair vec2 = FeatureMatrix::VectorPair(std::make_shared<FeatureVector>(set2, basis), outdim-1);

  mat = std::make_shared<FeatureMatrix>(std::vector<FeatureMatrix::VectorPair>{vec1, vec2}, domain);
  EXPECT_EQ(mat->numBasisFunctions, set1->NumIndices() + (outdim-1)*set2->NumIndices());
  EXPECT_EQ(mat->InputDimension(), indim);
  EXPECT_EQ(mat->numFeatureVectors, outdim);

  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);
  const Eigen::VectorXd y = domain->MapToHypercube(x);

  EXPECT_NEAR((mat->GetFeatureVector(0)->Evaluate(y)-vec1.first->Evaluate(y)).norm(), 0.0, 1.0e-14);
  for( std::size_t i=1; i<outdim; ++i ) { EXPECT_NEAR((mat->GetFeatureVector(i)->Evaluate(y)-vec2.first->Evaluate(y)).norm(), 0.0, 1.0e-14); }

  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(mat->numBasisFunctions);
  const Eigen::VectorXd output = mat->ApplyTranspose(x, coeff);
  EXPECT_EQ(output.size(), outdim);

  Eigen::VectorXd expected(outdim);
  expected(0) = vec1.first->Evaluate(y).dot(coeff.head(set1->NumIndices()));
  const Eigen::VectorXd eval2 = vec2.first->Evaluate(y);  
  for( std::size_t i=0; i<outdim-1; ++i ) { expected(i+1) = eval2.dot(coeff.segment(set1->NumIndices() + i*set2->NumIndices(), set2->NumIndices())); }
  EXPECT_NEAR((output-expected).norm(), 0.0, 1.0e-14);

  TestDerivatives({vec1.first, vec2.first, vec2.first});
}
  
} // namespace tests
} // namespace clf

