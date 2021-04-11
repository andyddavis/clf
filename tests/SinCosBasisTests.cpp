#include <gtest/gtest.h>

#include "clf/SinCosBasis.hpp"

namespace pt = boost::property_tree;
using namespace clf;

TEST(SinCosBasisTests, Construct) {
  pt::ptree pt;
  pt.put("Type", "SinCosBasis");

  auto sincosBasis = BasisFunctions::Construct(pt);
}
