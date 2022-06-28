#include <gtest/gtest.h>

#include "clf/MultiIndex.hpp"

using namespace clf;

TEST(MultiIndexTests, BasicTest) {
  const std::size_t d = 5;
  std::vector<std::size_t> alpha(d);
  for( std::size_t i=0; i<d; ++i ) { alpha[i] = rand()%13; }

  MultiIndex ind(alpha);
  EXPECT_EQ(ind.Dimension(), d);
}
