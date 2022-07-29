#include <gtest/gtest.h>

#include "clf/MultiIndex.hpp"

using namespace clf;

TEST(MultiIndexTests, BasicTest) {
  const std::size_t d = 5;
  std::vector<std::size_t> alpha(d);
  std::size_t order = 0;
  for( std::size_t i=0; i<d; ++i ) { 
    alpha[i] = rand()%13; 
    order += alpha[i];
  }

  MultiIndex ind(alpha);
  EXPECT_EQ(ind.Dimension(), d);
  EXPECT_EQ(ind.Order(), order);
}
