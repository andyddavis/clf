#include <gtest/gtest.h>

#include <MUQ/Modeling/Distributions/Gaussian.h>

#include "clf/LinearModel.hpp"
#include "clf/SupportPointSampler.hpp"
#include "clf/PolynomialBasis.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;

namespace clf {
namespace tests {

/// A class that runs the tests for clf::SupportPointSampler
class SupportPointSamplerTests : public::testing::Test {
protected:
  /// Set up information to test the support point sampler
  virtual void SetUp() override {
    // we will sample points from a Gaussian distribution
    randVar = std::make_shared<Gaussian>(indim)->AsVariable();

    // the order of the total order basis
    const std::size_t order = 2;
    pt.put("SupportPoint.BasisFunctions", "Basis1, Basis2");
    pt.put("SupportPoint.Basis1.Type", "TotalOrderPolynomials");
    pt.put("SupportPoint.Basis1.Order", order);
    pt.put("SupportPoint.Basis2.Type", "TotalOrderPolynomials");
    pt.put("SupportPoint.Basis2.Order", order);
  }

  /// Check the support point sampler
  virtual void TearDown() override {
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

  /// The input dimension
  const std::size_t indim = 3;

  /// The output dimension
  const std::size_t outdim = 2;

  /// The support points are sampled from this distribution 
  std::shared_ptr<RandomVariable> randVar;

  /// The options for the support point sampler
  pt::ptree pt;

  /// The support point sampler
  std::shared_ptr<SupportPointSampler> sampler;
};

TEST_F(SupportPointSamplerTests, SamplePointsDefaultModel) {
  pt.put("OutputDimension", outdim);
  sampler = std::make_shared<SupportPointSampler>(randVar, pt);
}

TEST_F(SupportPointSamplerTests, SamplePointsCustomModel) {
  // create a model
  pt::ptree modelOptions;
  modelOptions.put("InputDimension", indim);
  modelOptions.put("OutputDimension", outdim);
  auto model = std::make_shared<LinearModel>(modelOptions);

  // the support point sampler
  sampler = std::make_shared<SupportPointSampler>(randVar, model, pt);
}

} // namespace tests
} // namespace clf
