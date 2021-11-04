#include <gtest/gtest.h>

#include <MUQ/Modeling/Distributions/Gaussian.h>

#include "clf/LinearModel.hpp"
#include "clf/CollocationPointSampler.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace clf;

TEST(CollocationPointSamplerTests, Construction) {
  // the input and ouptut dimesnions
  const std::size_t indim = 6, outdim = 5;

  // the distribution we sample the colocation points from
  auto dist = std::make_shared<Gaussian>(indim)->AsVariable();

  // the model we wish to solve
  pt::ptree modelOptions;
  modelOptions.put("InputDimension", indim);
  modelOptions.put("OutputDimension", outdim);
  auto model = std::make_shared<LinearModel>(modelOptions);

  auto sampler = std::make_shared<CollocationPointSampler>(dist, model);

  Eigen::VectorXd mean = Eigen::VectorXd::Zero(indim);
  const std::size_t n = 2.0e5;
  for( std::size_t i=0; i<n; ++i ) {
    auto pnt = sampler->Sample(i, n);
    mean += pnt->x/n;
    EXPECT_EQ(pnt->model, model);
  }
  EXPECT_NEAR(mean.norm(), 0.0, 1.0e-2);
}
