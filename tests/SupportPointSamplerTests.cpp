#include <gtest/gtest.h>

#include <MUQ/Modeling/Distributions/Gaussian.h>

#include "clf/LinearModel.hpp"
#include "clf/SupportPointSampler.hpp"
#include "clf/PolynomialBasis.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace clf;

TEST(SupportPointSamplerTests, SamplePoints) {
  const std::size_t indim = 3, outdim = 2; // the in/output dimension

  // create a model
  pt::ptree modelOptions;
  modelOptions.put("InputDimension", indim);
  modelOptions.put("OutputDimension", outdim);
  auto model = std::make_shared<LinearModel>(modelOptions);

  // we will sample points from a Gaussian distribution
  auto randVar = std::make_shared<Gaussian>(indim)->AsVariable();

  // the order of the total order basis
  const std::size_t order = 2;

  pt::ptree pt;
  pt.put("SupportPoint.BasisFunctions", "Basis1, Basis2");
  pt.put("SupportPoint.Basis1.Type", "TotalOrderPolynomials");
  pt.put("SupportPoint.Basis1.Order", order);
  pt.put("SupportPoint.Basis2.Type", "TotalOrderPolynomials");
  pt.put("SupportPoint.Basis2.Order", order);

  // the support point sampler
  auto sampler = std::make_shared<SupportPointSampler>(randVar, model, pt);

  const std::size_t npoints = 15;
  for( std::size_t i=0; i<npoints; ++i ) {
    auto point = sampler->Sample();
    EXPECT_TRUE(point);

    const std::vector<std::shared_ptr<const BasisFunctions> >& bases = point->GetBasisFunctions();
    EXPECT_EQ(point->NumNeighbors(), 11);

    EXPECT_EQ(bases.size(), outdim);
    for( const auto& it : bases ) {
      EXPECT_TRUE(it);
      auto pointBasis = std::dynamic_pointer_cast<const SupportPointBasis>(it);
      EXPECT_TRUE(pointBasis);
      EXPECT_TRUE(std::dynamic_pointer_cast<const PolynomialBasis>(pointBasis->basis));
    }
  }
}
