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
    if( outind==0 ) {
      // a sin/cos function
      return std::sin(M_PI*x(0))*std::cos(2.0*M_PI*x(2)) + std::cos(M_PI*x(2));
    } else if( outind==1 ) {
      // a quadratic function
      return x(1)*x(0) + x(0) + 1.0;
    } else {
      return 0.0;
    }
  }
private:
};

TEST(LocalFunctionTests, Construction) {
  // the input and output dimensions
  const std::size_t indim = 3, outdim = 2;

  pt::ptree modelOptions;
  modelOptions.put("InputDimension", indim);
  modelOptions.put("OutputDimension", outdim);
  auto model = std::make_shared<ExampleModelForLocalFunctionTests>(modelOptions);

  // the order of the total order polynomial and sin/cos bases
  const std::size_t orderPoly = 5, orderSinCos = 2;

  // options for the support point
  pt::ptree suppOptions;
  suppOptions.put("NumNeighbors", 75);
  suppOptions.put("BasisFunctions", "Basis1, Basis2");
  //suppOptions.put("Basis1.Type", "TotalOrderSinCos");
  //suppOptions.put("Basis1.Order", orderSinCos);
  suppOptions.put("Basis1.Type", "TotalOrderPolynomials");
  suppOptions.put("Basis1.Order", orderPoly);
  suppOptions.put("Basis2.Type", "TotalOrderPolynomials");
  suppOptions.put("Basis2.Order", orderPoly);

  // create a support point cloud so that this point has nearest neighbors
  std::vector<std::shared_ptr<SupportPoint> > supportPoints(150);
  for( std::size_t i=0; i<supportPoints.size(); ++i ) { supportPoints[i] = SupportPoint::Construct(0.1*Eigen::VectorXd::Random(indim), model, suppOptions); }

  // create the support point cloud
  pt::ptree ptSupportPointCloud;
  auto cloud = SupportPointCloud::Construct(supportPoints, ptSupportPointCloud);

  // create the local function
  /*pt::ptree ptFunc;
  auto localFunction = std::make_shared<LocalFunction>(cloud, ptFunc);

  for( const auto& it : supportPoints ) {
    const Eigen::VectorXd eval = it->EvaluateLocalFunction(it->x);
    const Eigen::VectorXd expected = Eigen::Vector2d(std::sin(M_PI*it->x(0))*std::cos(2.0*M_PI*it->x(2)) + std::cos(M_PI*it->x(2)), it->x(1)*it->x(0) + it->x(0) + 1.0);

    std::cout << "eval: " << eval.transpose() << std::endl;
    std::cout << "expected: " << expected.transpose() << std::endl;
    std::cout << std::endl;
  }*/
}
