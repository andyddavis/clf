#include <gtest/gtest.h>

#include "clf/LegendrePolynomials.hpp"
#include "clf/FeatureMatrix.hpp"

using namespace clf;

TEST(FeatureMatrixTests, SingleFeatureVectorTest) {
  const std::size_t indim = 5;
  const std::size_t outdim = 3;
  const std::size_t maxOrder = 4;
  std::shared_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(indim, maxOrder);
  
  auto basis = std::make_shared<LegendrePolynomials>();
  
  auto vec = std::make_shared<FeatureVector>(set, basis);

  FeatureMatrix mat(vec, outdim);
  EXPECT_EQ(mat.numBasisFunctions, outdim*set->NumIndices());
  EXPECT_EQ(mat.numFeatureVectors, outdim);

  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);
  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(mat.numBasisFunctions);
  const Eigen::VectorXd output = mat.ApplyTranspose(x, coeff);
  EXPECT_EQ(output.size(), outdim);

  Eigen::VectorXd expected(outdim);
  const Eigen::VectorXd eval = vec->Evaluate(x);
  for( std::size_t i=0; i<outdim; ++i ) { expected(i) = eval.dot(coeff.segment(i*set->NumIndices(), set->NumIndices())); }
  EXPECT_NEAR((output-expected).norm(), 0.0, 1.0e-14);
}

TEST(FeatureMatrixTests, MultiFeatureVectorTest) {
  const std::size_t indim = 5;
  const std::size_t outdim = 2;
  const std::size_t maxOrder1 = 4;
  std::shared_ptr<MultiIndexSet> set1 = MultiIndexSet::CreateTotalOrder(indim, maxOrder1);
  const std::size_t maxOrder2 = 8;
  std::shared_ptr<MultiIndexSet> set2 = MultiIndexSet::CreateTotalOrder(indim, maxOrder2);
  
  auto basis = std::make_shared<LegendrePolynomials>();
  
  auto vec1 = std::make_shared<FeatureVector>(set1, basis);
  auto vec2 = std::make_shared<FeatureVector>(set2, basis);

  FeatureMatrix mat({vec1, vec2});
  EXPECT_EQ(mat.numBasisFunctions, set1->NumIndices() + set2->NumIndices());
  EXPECT_EQ(mat.numFeatureVectors, outdim);

  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);
  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(mat.numBasisFunctions);
  const Eigen::VectorXd output = mat.ApplyTranspose(x, coeff);
  EXPECT_EQ(output.size(), outdim);

  Eigen::VectorXd expected(outdim);
  expected(0) = vec1->Evaluate(x).dot(coeff.head(set1->NumIndices()));
  expected(1) = vec2->Evaluate(x).dot(coeff.tail(set2->NumIndices()));
  EXPECT_NEAR((output-expected).norm(), 0.0, 1.0e-14);
}

TEST(FeatureMatrixTests, RepeatedFeatureVectorsTest) {
  const std::size_t indim = 5;
  const std::size_t outdim = 3;
  const std::size_t maxOrder1 = 4;
  std::shared_ptr<MultiIndexSet> set1 = MultiIndexSet::CreateTotalOrder(indim, maxOrder1);
  const std::size_t maxOrder2 = 8;
  std::shared_ptr<MultiIndexSet> set2 = MultiIndexSet::CreateTotalOrder(indim, maxOrder2);
  
  auto basis = std::make_shared<LegendrePolynomials>();
  
  const FeatureMatrix::VectorPair vec1 = FeatureMatrix::VectorPair(std::make_shared<FeatureVector>(set1, basis), 1);
  const FeatureMatrix::VectorPair vec2 = FeatureMatrix::VectorPair(std::make_shared<FeatureVector>(set2, basis), outdim-1);

  FeatureMatrix mat({vec1, vec2});
  EXPECT_EQ(mat.numBasisFunctions, set1->NumIndices() + (outdim-1)*set2->NumIndices());
  EXPECT_EQ(mat.numFeatureVectors, outdim);

  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);
  const Eigen::VectorXd coeff = Eigen::VectorXd::Random(mat.numBasisFunctions);
  const Eigen::VectorXd output = mat.ApplyTranspose(x, coeff);
  EXPECT_EQ(output.size(), outdim);

  Eigen::VectorXd expected(outdim);
  expected(0) = vec1.first->Evaluate(x).dot(coeff.head(set1->NumIndices()));
  const Eigen::VectorXd eval2 = vec2.first->Evaluate(x);  
  for( std::size_t i=0; i<outdim-1; ++i ) { expected(i+1) = eval2.dot(coeff.segment(set1->NumIndices() + i*set2->NumIndices(), set2->NumIndices())); }
  EXPECT_NEAR((output-expected).norm(), 0.0, 1.0e-14);
}
