#include <gtest/gtest.h>

#include "clf/LocalFunction.hpp"

namespace pt = boost::property_tree;
using namespace clf;

class ExampleModelForLocalFunctionTests : public Model {
public:

  inline ExampleModelForLocalFunctionTests(pt::ptree const& pt) : Model(pt) {}

  virtual ~ExampleModelForLocalFunctionTests() = default;

protected:

  inline virtual double RightHandSideComponentImpl(Eigen::VectorXd const& x, std::size_t const outind) const override {
    // a quadratic function
    return x(0)*x(0) + x(0) + 1.0;
  }
private:
};

TEST(LocalFunctionTests, Construction) {
  /*// evenly spaced point locations
  const Eigen::VectorXd points = Eigen::VectorXd::LinSpaced(10, 0.0, 1.0);

  pt::ptree ptSupportPoints;
  ptSupportPoints.put("BasisFunctions", "Basis");
  ptSupportPoints.put("Basis.Type", "TotalOrderPolynomials");
  ptSupportPoints.put("Basis.Order", 2);

  pt::ptree modelOptions;
  modelOptions.put("InputDimension", 1);
  modelOptions.put("OutputDimension", 1);
  auto model = std::make_shared<ExampleModelForLocalFunctionTests>(modelOptions);

  // create the support points
  std::vector<std::shared_ptr<SupportPoint> > supportPoints(points.size());
  for( std::size_t i=0; i<points.size(); ++i ) { supportPoints[i] = SupportPoint::Construct(points.row(i), model, ptSupportPoints); }

  // create the support point cloud
  pt::ptree ptSupportPointCloud;
  auto cloud = std::make_shared<SupportPointCloud>(supportPoints, ptSupportPointCloud);

  // create the local function
  pt::ptree ptFunc;
  auto localFunction = std::make_shared<LocalFunction>(cloud, ptFunc);*/
}
