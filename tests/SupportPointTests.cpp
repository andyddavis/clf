#include <gtest/gtest.h>

#include "clf/SupportPoint.hpp"

using namespace clf;

TEST(SupportPointTests, Construction) {
  auto point = std::make_shared<SupportPoint>();
}
