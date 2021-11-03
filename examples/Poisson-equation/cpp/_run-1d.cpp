#include <iostream>

#include <MUQ/Modeling/Distributions/UniformBox.h>

#include <clf/SupportPointCloud.hpp>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace clf;

int main(int argc, char **argv) {
  // create the random variable that we use to sample support points
  auto rv = std::make_shared<UniformBox>(0.0, 1.0)->AsVariable();

  std::size_t order = 2;

  // create the support point sampler 
  pt::ptree ptSampler;
  ptSampler.put("SupportPoint.BasisFunctions", "Basis");
  ptSampler.put("SupportPoint.Basis.Type", "TotalOrderPolynomials");
  ptSampler.put("SupportPoint.Basis.Order", order);
  ptSampler.put("OutputDimension", 1);
  auto sampler = std::make_shared<SupportPointSampler>(rv, ptSampler);

  // create a support point cloud
  pt::ptree ptSupportCloud;
  ptSupportCloud.put("NumSupportPoints", 10);
  auto supportCloud = SupportPointCloud::Construct(sampler, ptSupportCloud);

  // write the support points to file 
  const std::string file = "examples/Poisson-equation/cpp/Poisson-1d.h5";
  supportCloud->WriteToFile(file);
}
