#include <gtest/gtest.h>

#include "clf/GlobalCost.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace clf;

class ExampleModelForGlobalCostTests : public Model {
public:

  inline ExampleModelForGlobalCostTests(pt::ptree const& pt) : Model(pt) {}

  virtual ~ExampleModelForGlobalCostTests() = default;

protected:

  /**
  @param[in] x The point \f$x \in \Omega \f$
  \return The evaluation of \f$f(x)\f$
  */
  inline virtual Eigen::VectorXd RightHandSideVectorImpl(Eigen::VectorXd const& x) const override {
    return Eigen::Vector2d(
      std::sin(2.0*M_PI*x(0))*std::cos(M_PI*x(1)) + std::cos(x(0)),
      x.prod()
    );
  }

private:
};

class GlobalCostTests : public::testing::Test {
public:
  /// Set up information to test the support point
  virtual void SetUp() override {
    pt::ptree modelOptions;
    modelOptions.put("InputDimension", indim);
    modelOptions.put("OutputDimension", outdim);

    // the order of the total order polynomial and sin/cos bases
    const std::size_t orderPoly = 2, orderSinCos = 1;

    const std::size_t npoints = 3;

    // options for the support point
    pt::ptree suppOptions;
    suppOptions.put("NumNeighbors", npoints*npoints+1);
    suppOptions.put("CoupledScale", coupledScale);
    suppOptions.put("BasisFunctions", "Basis1, Basis2");
    suppOptions.put("Basis1.Type", "TotalOrderSinCos");
    suppOptions.put("Basis1.Order", orderSinCos);
    suppOptions.put("Basis1.LocalBasis", false);
    suppOptions.put("Basis2.Type", "TotalOrderPolynomials");
    suppOptions.put("Basis2.Order", orderPoly);
    point = SupportPoint::Construct(
      Eigen::VectorXd::Random(indim),
      std::make_shared<ExampleModelForGlobalCostTests>(modelOptions),
      suppOptions);

    // create a support point cloud so that this point has nearest neighbors
    supportPoints.resize(4*npoints*npoints+1);
    supportPoints[0] = point;
    // add points on a grid so we know that they are well-poised---make sure there is an even number of points on each side so that the center point is not on the grid
    for( std::size_t i=0; i<2*npoints; ++i ) {
      for( std::size_t j=0; j<2*npoints; ++j ) {
        supportPoints[2*npoints*i+j+1] = SupportPoint::Construct(
          point->x+0.1*Eigen::Vector2d((double)i/(2*npoints-1)-0.5, (double)j/(2*npoints-1)-0.5),
          std::make_shared<ExampleModelForGlobalCostTests>(modelOptions),
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

TEST_F(GlobalCostTests, CostEvaluationAndDerivatives) {
  // create the global cost
  pt::ptree costOptions;
  auto cost = std::make_shared<GlobalCost>(cloud, costOptions);
  EXPECT_EQ(cost->inputSizes(0), cloud->numCoefficients);

  // choose random coefficients
  const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(cost->inputSizes(0));

  // set these coefficients at each local point
  std::size_t ind = 0;
  for( const auto& point : supportPoints ) {
    // extract the coefficients associated with point
    Eigen::Map<const Eigen::VectorXd> coeff(&coefficients(ind), point->NumCoefficients());
    ind += point->NumCoefficients();

    point->Coefficients() = coeff;
  }

  // compute the global cost
  const double cst = cost->Cost(coefficients);
  double expectedCost = 0.0;
  for( const auto& point : supportPoints ) {
    expectedCost += point->ComputeUncoupledCost();
    expectedCost += point->ComputeCoupledCost();
  }
  EXPECT_NEAR(cst, expectedCost, 1.0e-10);

  // compute the gradient
  const Eigen::VectorXd gradFD = cost->GradientByFD(0, 0, ref_vector<Eigen::VectorXd>(1, coefficients), 0.75*Eigen::VectorXd::Ones(1));
  const Eigen::VectorXd grad = cost->Gradient(0, std::vector<Eigen::VectorXd>(1, coefficients), (0.75*Eigen::VectorXd::Ones(1)).eval());
  EXPECT_NEAR((grad-gradFD).norm()/gradFD.norm(), 0.0, 1.0e-5);
}
