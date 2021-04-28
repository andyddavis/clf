#include <gtest/gtest.h>

#include "clf/CoupledCost.hpp"
#include "clf/SupportPointCloud.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::Optimization;
using namespace clf;

class ExampleModelForCoupledCostTests : public Model {
public:

  inline ExampleModelForCoupledCostTests(pt::ptree const& pt) : Model(pt) {}

  virtual ~ExampleModelForCoupledCostTests() = default;

protected:

private:
};

class CoupledCostTests : public::testing::Test {
public:
  /// Set up information to test the support point
  virtual void SetUp() override {}

  /// Create the support point cloud given optimization opertions
  void CreateCloud(pt::ptree const& optimization = pt::ptree()) {
    pt::ptree modelOptions;
    modelOptions.put("InputDimension", indim);
    modelOptions.put("OutputDimension", outdim);

    // the order of the total order polynomial and sin/cos bases
    const std::size_t orderPoly = 4, orderSinCos = 2;

    const std::size_t npoints = 6;

    // options for the support point
    pt::ptree suppOptions;
    suppOptions.put("NumNeighbors", npoints*npoints+1);
    suppOptions.put("BasisFunctions", "Basis1, Basis2");
    suppOptions.put("Basis1.Type", "TotalOrderSinCos");
    suppOptions.put("Basis1.Order", orderSinCos);
    suppOptions.put("Basis1.LocalBasis", false);
    suppOptions.put("Basis2.Type", "TotalOrderPolynomials");
    suppOptions.put("Basis2.Order", orderPoly);
    suppOptions.add_child("Optimization", optimization);
    point = SupportPoint::Construct(
      Eigen::VectorXd::Random(indim),
      std::make_shared<ExampleModelForCoupledCostTests>(modelOptions),
      suppOptions);

    // create a support point cloud so that this point has nearest neighbors
    supportPoints.resize(4*npoints*npoints+1);
    supportPoints[0] = point;
    // add points on a grid so we know that they are well-poised---make sure there is an even number of points on each side so that the center point is not on the grid
    for( std::size_t i=0; i<2*npoints; ++i ) {
      for( std::size_t j=0; j<2*npoints; ++j ) {
        supportPoints[2*npoints*i+j+1] = SupportPoint::Construct(
          point->x+0.1*Eigen::Vector2d((double)i/(2*npoints-1)-0.5, (double)j/(2*npoints-1)-0.5),
          std::make_shared<ExampleModelForCoupledCostTests>(modelOptions),
          suppOptions);
      }
    }
    pt::ptree ptSupportPointCloud;
    cloud = SupportPointCloud::Construct(supportPoints, ptSupportPointCloud);
  }

  /// Make sure everything is what we expect
  virtual void TearDown() override {}

  /// The input and output dimensions
  const std::size_t indim = 2, outdim = 2;

  std::vector<std::shared_ptr<SupportPoint> > supportPoints;

  std::shared_ptr<SupportPoint> point;

  std::shared_ptr<SupportPointCloud> cloud;

  const double coupledScale = 0.25;
private:
};

TEST_F(CoupledCostTests, CostEvaluationAndDerivatives) {
  CreateCloud();

  // create the uncoupled cost for each neighbor
  for( auto it=supportPoints.begin(); it!=supportPoints.end(); ++it ) {
    pt::ptree costOptions;
    costOptions.put("CoupledScale", coupledScale);
    auto cost = std::make_shared<CoupledCost>(point, *it, costOptions);
    EXPECT_EQ(cost->inputSizes(0), point->NumCoefficients()+(*it)->NumCoefficients());
    EXPECT_DOUBLE_EQ(cost->coupledScale, coupledScale);
    EXPECT_EQ(cost->Coupled(), (*it)!=point & point->IsNeighbor((*it)->GlobalIndex()));

    // choose random coefficients
    const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(cost->inputSizes(0));

    // compute the coupling cost
    const double cst = cost->Cost(coefficients);
    if( !cost->Coupled() ) {
      EXPECT_DOUBLE_EQ(cst, 0.0);
    } else {
      const Eigen::VectorXd pntEval = point->EvaluateLocalFunction((*it)->x, coefficients.head(point->NumCoefficients()));
      const Eigen::VectorXd neighEval = (*it)->EvaluateLocalFunction((*it)->x, coefficients.tail((*it)->NumCoefficients()));
      const Eigen::VectorXd diff = pntEval - neighEval;
      EXPECT_NEAR(cst, coupledScale*diff.dot(diff)*point->NearestNeighborKernel(point->LocalIndex((*it)->GlobalIndex()))/2.0, 1.0e-10);
    }

    // compute the coupling gradient
    const Eigen::VectorXd grad = cost->Gradient(0, std::vector<Eigen::VectorXd>(1, coefficients), (0.75*Eigen::VectorXd::Ones(1)).eval());
    EXPECT_EQ(grad.size(), coefficients.size());
    if( !cost->Coupled() ) {
      EXPECT_DOUBLE_EQ(grad.norm(), 0.0);
    } else {
      const Eigen::VectorXd gradFD = cost->GradientByFD(0, 0, ref_vector<Eigen::VectorXd>(1, coefficients), 0.75*Eigen::VectorXd::Ones(1));
      EXPECT_NEAR((gradFD-grad).norm()/gradFD.norm(), 0.0, 1.0e-6);
    }
  }
}
