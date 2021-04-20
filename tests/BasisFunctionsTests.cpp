#include <gtest/gtest.h>

#include "clf/BasisFunctions.hpp"

namespace pt = boost::property_tree;
using namespace muq::Utilities;
using namespace clf;

TEST(BasisFunctionsTests, Construction) {
  // the input dimension
  const std::size_t dim = 5;

  // the multi-index set
  auto multis = std::make_shared<MultiIndexSet>(dim);

  // an invalid basis function name to test exception handling
  const std::string name = "ALongInvalidBasisNameThatNoOneShouldUseBecauseItIsForTesting";
  pt::ptree pt;
  pt.put("Type", name);

  try {
    auto basis = BasisFunctions::Construct(multis, pt);
  } catch( exceptions::BasisFunctionsNameConstuctionException const& exc ) {
    EXPECT_EQ(exc.basisName, name);
  }
}
