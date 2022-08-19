#include <gtest/gtest.h>

#include "clf/LinearDifferentialOperator.hpp"

using namespace clf;

TEST(LinearDifferentialOperator, AllSame) {
  std::size_t indim = 8;
  std::size_t outdim = 5;
  Eigen::MatrixXi counts = Eigen::MatrixXi::Zero(indim, 3);
  for( std::size_t i=0; i<counts.cols(); ++i ) {
    for( std::size_t j=0; j<9; ++j ) { ++counts(rand()%(indim-1), i); }
  }
  
  auto linOp = std::make_shared<LinearDifferentialOperator>(counts, outdim);
  EXPECT_EQ(linOp->outdim, outdim);
  EXPECT_EQ(linOp->indim, indim);
  EXPECT_EQ(linOp->NumOperators(), counts.cols());

  for( std::size_t i=0; i<outdim; ++i ) {
    const LinearDifferentialOperator::CountPair C = linOp->Counts(i);
    EXPECT_EQ(C.second, outdim);
    EXPECT_EQ(C.first.rows(), indim);
    EXPECT_EQ(C.first.cols(), counts.cols());
    for( std::size_t s=0; s<counts.cols(); ++s ) {
      for( std::size_t j=0; j<indim; ++j ) { EXPECT_EQ(C.first(j, s), counts(j, s)); }
    }
  }
}

TEST(LinearDifferentialOperator, AllDifferent) {
  std::size_t indim = 8;
  std::size_t outdim = 5;
  std::vector<Eigen::MatrixXi> counts(outdim);
  for( std::size_t i=0; i<outdim; ++i ) {
    counts[i] = Eigen::VectorXi::Zero(indim);
    for( std::size_t j=0; j<9; ++j ) { ++counts[i](rand()%(indim-1)); }
  }
  
  auto linOp = std::make_shared<LinearDifferentialOperator>(counts);
  EXPECT_EQ(linOp->outdim, outdim);
  EXPECT_EQ(linOp->indim, indim);
  EXPECT_EQ(linOp->NumOperators(), 1);

  for( std::size_t i=0; i<outdim; ++i ) {
    const LinearDifferentialOperator::CountPair C = linOp->Counts(i);
    EXPECT_EQ(C.second, 1);
    EXPECT_EQ(C.first.size(), indim);
    for( std::size_t j=0; j<indim; ++j ) { EXPECT_EQ(C.first(j), counts[i](j)); }
  }
}

TEST(LinearDifferentialOperator, Mixed) {
  std::size_t indim = 8;
  std::size_t outdim = 5;
  std::vector<LinearDifferentialOperator::CountPair> counts(3);
  for( std::size_t i=0; i<counts.size(); ++i ) {
    counts[i] = LinearDifferentialOperator::CountPair(Eigen::VectorXi::Zero(indim), (i==1? 1 : 2));
    for( std::size_t j=0; j<9; ++j ) { ++counts[i].first(rand()%(indim-1)); }
  }
  
  auto linOp = std::make_shared<LinearDifferentialOperator>(counts);
  EXPECT_EQ(linOp->outdim, outdim);
  EXPECT_EQ(linOp->indim, indim);
  EXPECT_EQ(linOp->NumOperators(), 1);

  std::size_t c = 0;
  for( std::size_t i=0; i<counts.size(); ++i ) {
    for( std::size_t s=0; s<counts[i].second; ++s ) {
      const LinearDifferentialOperator::CountPair C = linOp->Counts(c);
      EXPECT_EQ(C.second, counts[i].second);
      EXPECT_EQ(C.first.size(), indim);
      for( std::size_t j=0; j<indim; ++j ) { EXPECT_EQ(C.first(j), counts[i].first(j)); }

      ++c;
    }
  }
}