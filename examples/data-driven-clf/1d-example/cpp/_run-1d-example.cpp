#include <iostream>

#include <MUQ/Modeling/Distributions/UniformBox.h>

#include "clf/SupportPointCloud.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace clf;

int main(int argc, char **argv) {
  auto randVar = std::make_shared<UniformBox>(Eigen::RowVector2d(0.0, 2.0*M_PI))->AsVariable();

  // create a model
  pt::ptree modelOptions;
  modelOptions.put("InputDimension", 1);
  modelOptions.put("OutputDimension", 1);
  auto model = std::make_shared<Model>(modelOptions);

  // the order of the total order basis
  const std::size_t order = 2;

  pt::ptree samplerOptions;
  samplerOptions.put("SupportPoint.BasisFunctions", "Basis");
  samplerOptions.put("SupportPoint.Basis.Type", "TotalOrderPolynomials");
  samplerOptions.put("SupportPoint.Basis.Order", order);

  // the support point sampler
  auto sampler = std::make_shared<SupportPointSampler>(randVar, model, samplerOptions);

  // create the support point cloud
  pt::ptree cloudOptions;
  cloudOptions.put("NumSupportPoints", 100);
  auto cloud = SupportPointCloud::Construct(sampler, cloudOptions);

}
