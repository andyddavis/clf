#include <gtest/gtest.h>

#include "clf/MultiIndexSet.hpp"

using namespace clf;

TEST(MultiIndexSetTests, BasicTest) {
  // create a bunch of multi indices
  const std::size_t d = 5;
  const std::size_t q = 8;
  std::vector<std::unique_ptr<MultiIndex> > indices(q);
  for( std::size_t i=0; i<q; ++i ) {
    std::vector<std::size_t> alpha(d);
    for( std::size_t j=0; j<d; ++j ) { alpha[j] = rand()%13; }    
    indices[i] = std::make_unique<MultiIndex>(alpha);
  }

  // create the multi index set 
  MultiIndexSet set(indices);

  // check the number of indices 
  EXPECT_EQ(set.NumIndices(), q);

  // make sure the spatial dimension is correct 
  EXPECT_EQ(set.Dimension(), d);

  // make sure the max index is less than the expected number 
  for( std::size_t i=0; i<d; ++i ) { EXPECT_TRUE(set.MaxIndex(i)<13); }
}

TEST(MultiIndexSetTests, CreateTotalOrder) {
  const std::size_t dim = 3;
  const std::size_t maxOrder = 4;
  std::unique_ptr<MultiIndexSet> set = MultiIndexSet::CreateTotalOrder(dim, maxOrder);

  // useful functions for this test 
  auto fac = [&] (std::size_t const n) { 
    std::size_t r = 1; 
    for( std::size_t i=2; i<=n; ++i ) { r *= i; }
    return r;
  };
  auto choose = [&] (std::size_t const n, std::size_t const m) { return fac(n)/(fac(m)*fac(n-m)); };

  // make sure the total number of indices is what we expect
  std::size_t expectedNumIndices = 0;
  for( std::size_t i=0; i<=maxOrder; ++i ) { expectedNumIndices += choose(dim+i-1, i); }
  EXPECT_EQ(set->NumIndices(), expectedNumIndices);

  // make sure all of the indices are less than or equal to the max order
  for( const auto& it : set->indices ) { EXPECT_TRUE(it->Order()<=maxOrder); }

  // make sure the max index is the max order
  for( std::size_t i=0; i<dim; ++i ) { EXPECT_EQ(set->MaxIndex(i), maxOrder); }
}
