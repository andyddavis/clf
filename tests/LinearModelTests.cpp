#include <gtest/gtest.h>

#include "clf/LinearModel.hpp"

namespace pt = boost::property_tree;
using namespace clf;

TEST(LinearModelTests, Identity) {
  pt::ptree options;
  options.put("InputDimension", 3);
  options.put("OutputDimension", 3);
  //auto model = std::make_shared<LinearModel>(options);

  EXPECT_TRUE(false);
}
