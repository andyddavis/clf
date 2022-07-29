#include <gtest/gtest.h>

#include "clf/Parameters.hpp"

using namespace clf;

TEST(ParametersTest, BasicTest) {
  Parameters para;
  EXPECT_EQ(para.NumParameters(), 0);

  {
    const double a = 1.4;
    para.Add("A", a);
    EXPECT_EQ(para.NumParameters(), 1);
    
    const std::size_t b = 13;
    para.Add("B", b);
    EXPECT_EQ(para.NumParameters(), 2);
    
    const std::string c = "TEST";
    para.Add("C", c);
    EXPECT_EQ(para.NumParameters(), 3);
  }
  EXPECT_EQ(para.NumParameters(), 3);

  EXPECT_DOUBLE_EQ(para.Get<double>("A"), 1.4);
  EXPECT_EQ(para.Get<std::size_t>("B"), 13);
  EXPECT_PRED2([](std::string const& str1, std::string const& str2) { return str1==str2; }, para.Get<std::string>("C"), "TEST");
  // check default value
  EXPECT_EQ(para.Get("D", 10), 10);
}
