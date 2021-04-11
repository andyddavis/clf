#include <gtest/gtest.h>

#include "clf/BasisFunctions.hpp"

namespace pt = boost::property_tree;
using namespace clf;

TEST(BasisFunctionsTests, Construction) {
  const std::string name = "ALongInvalidBasisNameThatNoOneShouldUseBecauseItIsForTesting";
  pt::ptree pt;
  pt.put("Type", name);

  try {
    auto sincosBasis = BasisFunctions::Construct(pt);
  } catch( BasisFunctionsNameConstuctionException const& exc ) {
    EXPECT_EQ(exc.basisName, name);
  }
}
