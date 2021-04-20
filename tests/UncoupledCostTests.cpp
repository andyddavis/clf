#include <gtest/gtest.h>

#include "clf/UncoupledCost.hpp"
#include "clf/SupportPointCloud.hpp"

namespace pt = boost::property_tree;
using namespace muq::Optimization;
using namespace clf;

class ExampleIdentityModelForUncoupledCostTests : public Model {
public:

  inline ExampleIdentityModelForUncoupledCostTests(pt::ptree const& pt) : Model(pt) {}

  virtual ~ExampleIdentityModelForUncoupledCostTests() = default;

protected:

  /**
  @param[in] x The point \f$x \in \Omega \f$
  \return The evaluation of \f$f(x)\f$
  */
  inline virtual Eigen::VectorXd RightHandSideVectorImpl(Eigen::VectorXd const& x) const override { return Eigen::VectorXd::Constant(outputDimension, x.prod()); }

private:
};

TEST(UncoupledCostTests, Construction) {
  // the input and output dimensions
  const std::size_t indim = 4, outdim = 2;

  pt::ptree modelOptions;
  modelOptions.put("InputDimension", indim);
  modelOptions.put("OutputDimension", outdim);

  // the order of the total order polynomial and sin/cos bases
  const std::size_t orderPoly = 3, orderSinCos = 2;

  // options for the support point
  pt::ptree suppOptions;
  suppOptions.put("NumNeighbors", 75);
  suppOptions.put("BasisFunctions", "Basis1, Basis2");
  suppOptions.put("Basis1.Type", "TotalOrderSinCos");
  suppOptions.put("Basis1.Order", orderSinCos);
  suppOptions.put("Basis2.Type", "TotalOrderPolynomials");
  suppOptions.put("Basis2.Order", orderPoly);
  auto point = std::make_shared<SupportPoint>(
    Eigen::VectorXd::Random(indim),
    std::make_shared<ExampleIdentityModelForUncoupledCostTests>(modelOptions),
    suppOptions);

  // create a support point cloud so that this point has nearest neighbors
  std::vector<std::shared_ptr<SupportPoint> > supportPoints(75);
  supportPoints[0] = point;
  for( std::size_t i=1; i<supportPoints.size(); ++i ) {
    supportPoints[i] = std::make_shared<SupportPoint>(
      Eigen::VectorXd::Random(indim),
      std::make_shared<ExampleIdentityModelForUncoupledCostTests>(modelOptions),
      suppOptions);
  }
  pt::ptree ptSupportPointCloud;
  auto cloud = SupportPointCloud::Construct(supportPoints, ptSupportPointCloud);

  // create the uncoupled cost
  auto cost = std::make_shared<UncoupledCost>(point);
  EXPECT_EQ(cost->inputSizes(0), point->NumCoefficients());

  // the points should be the same
  auto costPt = cost->point.lock();
  EXPECT_NEAR((costPt->x-point->x).norm(), 0.0, 1.0e-10);

  // choose the vector of coefficients
  const Eigen::VectorXd coefficients = Eigen::VectorXd::Ones(point->NumCoefficients());

  // compute the true cost
  double trueCost = 0.0;
  {
    const Eigen::VectorXd kernel = point->NearestNeighborKernel();
    EXPECT_EQ(kernel.size(), supportPoints.size());
    for( std::size_t i=0; i<supportPoints.size(); ++i ) {
      const Eigen::VectorXd diff = point->Operator(supportPoints[i]->x, coefficients) - point->model->RightHandSide(supportPoints[i]->x);
      trueCost += kernel(i)*diff.dot(diff);
    }
    trueCost /= (2.0*supportPoints.size());
  }

  // compute the cost
  const double cst = cost->Cost(coefficients);
  EXPECT_NEAR(cst, trueCost, 1.0e-12);
}
