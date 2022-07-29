#include <gtest/gtest.h>

#include "clf/MultiIndexSet.hpp"

using namespace clf;

TEST(MultiIndexSetTests, BasicTest) {
  // create a bunch of multi indices
  const std::size_t d = 5;
  const std::size_t q = 8;
  std::vector<MultiIndex> indices;
  for( std::size_t i=0; i<q; ++i ) {
    std::vector<std::size_t> alpha(d);
    for( std::size_t j=0; j<d; ++j ) { alpha[j] = rand()%13; }    
    indices.emplace_back(alpha);
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

namespace clf { 
namespace tests {

/// A class to run tests for MultiIndexSet::CreateTotalOrder
class TotalOrderMultiIndexTests : public::testing::Test {
protected:
  /// Tear down the tests
  virtual void TearDown() override {
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
    for( const auto& it : set->indices ) { EXPECT_TRUE(it.Order()<=maxOrder); }
    
    // make sure the max index is the max order
    for( std::size_t i=0; i<dim; ++i ) { EXPECT_EQ(set->MaxIndex(i), maxOrder); }
  }
  
  /// The dimension of the multi-indices
  const std::size_t dim = 3;

  /// The maximum order
  const std::size_t maxOrder = 4;

  /// The set of multi-indices
  std::unique_ptr<MultiIndexSet> set;
};

TEST_F(TotalOrderMultiIndexTests, BasicConstruction) {
  set = MultiIndexSet::CreateTotalOrder(dim, maxOrder);
}

TEST_F(TotalOrderMultiIndexTests, ParameterConstruction) {
  auto para = std::make_shared<Parameters>();
  para->Add("InputDimension", dim);
  para->Add("MaximumOrder", maxOrder);

  set = MultiIndexSet::CreateTotalOrder(para);
}

} // namespace tests
} // namespace clf 
