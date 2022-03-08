#include <gtest/gtest.h>

#include "clf/LinearModel.hpp"
#include "clf/CoupledSupportPoint.hpp"
#include "clf/LocalFunctions.hpp"

#include "TestModels.hpp"

namespace pt = boost::property_tree;

namespace clf {
namespace tests {
/// A class to run the tests for clf::LocalFunctions
class LocalFunctionsTests : public::testing::Test {
protected:
  /// Create the support point cloud given the coupling scale
  /**
  @param[in] couplingScale The coupling scale. If the coupling scale is greater than zero, this problem minimizes clf::GlobalCost. If the coupling scale is zero, this problem minimizes the clf::UncoupledCost for each support point.
  */
  inline void CreateSupportPointCloud(double const couplingScale = 0.0) {
    pt::ptree modelOptions;
    modelOptions.put("InputDimension", indim);
    modelOptions.put("OutputDimension", outdim);
    auto model = std::make_shared<TwoDimensionalAlgebraicModel>(modelOptions);

    // the order of the total order polynomial and sin/cos bases
    const std::size_t orderPoly = 3, orderSinCos = 2;

    const std::size_t npoints = 5;

    // options for the support point
    pt::ptree suppOptions;
    suppOptions.put("NumNeighbors", npoints*npoints);
    suppOptions.put("CoupledScale", couplingScale);
    suppOptions.put("BasisFunctions", "Basis1, Basis2");
    suppOptions.put("Basis1.Type", "TotalOrderSinCos");
    suppOptions.put("Basis1.Order", orderSinCos);
    suppOptions.put("Basis1.LocalBasis", false);
    suppOptions.put("Basis2.Type", "TotalOrderPolynomials");
    suppOptions.put("Basis2.Order", orderPoly);

    // add points on a grid
    std::vector<std::shared_ptr<SupportPoint> > supportPoints(4*npoints*npoints);
    for( std::size_t i=0; i<2*npoints; ++i ) {
      for( std::size_t j=0; j<2*npoints; ++j ) {
        supportPoints[2*npoints*i+j] = SupportPoint::Construct(
          0.1*Eigen::Vector2d((double)i/(2*npoints-1)-0.5, (double)j/(2*npoints-1)-0.5),
          model, suppOptions);
      }
    }

    // create the support point cloud
    pt::ptree ptSupportPointCloud;
    cloud = SupportPointCloud::Construct(supportPoints, ptSupportPointCloud);
  }

  /// Make sure everything is what we expect
  virtual void TearDown() override {
    // the cost of the optimial coefficients
    const double cost = func->CoefficientCost();
    EXPECT_TRUE(cost<1.0e-4);

    for( std::size_t i=0; i<cloud->NumPoints(); ++i ) {
      auto it = cloud->GetSupportPoint(i);

      const Eigen::VectorXd eval = it->EvaluateLocalFunction(it->x);
      const Eigen::VectorXd expected = Eigen::Vector2d(std::sin(2.0*M_PI*it->x(0))*std::cos(M_PI*it->x(1)) + std::cos(M_PI*it->x(1)), it->x(1)*it->x(0));
      EXPECT_EQ(eval.size(), expected.size());
      EXPECT_NEAR((eval-expected).norm(), 0.0, 10.0*std::sqrt(cost));
    }

    for( std::size_t i=0; i<10; ++i ) {
      // pick a random point
      const Eigen::VectorXd x = 0.01*Eigen::VectorXd::Random(indim);

      // find the nearest support point and the squared distance to it
      std::size_t ind; double dist;
      std::tie(ind, dist) = func->NearestNeighbor(x);
      for( std::size_t i=0; i<cloud->NumPoints(); ++i ) {
	auto it = cloud->GetSupportPoint(i);
	EXPECT_TRUE(dist<=(x-it->x).dot(x-it->x)+1.0e-10);
      }

      // evaluate the support point
      const Eigen::VectorXd eval = func->Evaluate(x);
      const Eigen::VectorXd expected = Eigen::Vector2d(std::sin(2.0*M_PI*x(0))*std::cos(M_PI*x(1)) + std::cos(M_PI*x(1)), x(1)*x(0));
      EXPECT_EQ(eval.size(), expected.size());
      EXPECT_NEAR((eval-expected).norm(), 0.0, 10.0*std::sqrt(cost));
    }
  }

  /// The cloud containing all of the support points
  std::shared_ptr<SupportPointCloud> cloud;

  /// The local function we are testing
  std::shared_ptr<LocalFunctions> func;

  /// The input dimension
  const std::size_t indim = 2;

  /// The output dimension
  const std::size_t outdim = 2;
};

TEST_F(LocalFunctionsTests, UncoupledComputation_LevenbergMarquardt) {
  CreateSupportPointCloud();

  // create the local function
  pt::ptree ptFunc;
  func = std::make_shared<LocalFunctions>(cloud, ptFunc);

  pt::ptree optimizationOptions;
  optimizationOptions.put("Method", "LevenbergMarquardt");
  optimizationOptions.put("InitialDamping", 1.0);
  optimizationOptions.put("FunctionTolerance", 1.0e-4);
  optimizationOptions.put("NumThreads", 5);
  const double cost = func->ComputeOptimalCoefficients(optimizationOptions);
  EXPECT_TRUE(cost<1.0e-4);
}

/*TEST_F(LocalFunctionsTests, UncoupledComputation_NLopt) {
  CreateSupportPointCloud();

  // create the local function
  pt::ptree ptFunc;
  func = std::make_shared<LocalFunctions>(cloud, ptFunc);

  pt::ptree optimizationOptions;
  optimizationOptions.put("Method", "NLopt");
  optimizationOptions.put("FunctionTolerance", 1.0e-4);
  optimizationOptions.put("NumThreads", 5);
  const double cost = func->ComputeOptimalCoefficients(optimizationOptions);
  EXPECT_TRUE(cost<1.0e-4);
}*/

/*TEST(LocalFunctionTests, UncoupledLinearModel) {
  // each model is a linear model
  pt::ptree modelOptions;
  modelOptions.put("InputDimension", 2);
  modelOptions.put("OutputDimension", 2);
  auto model = std::make_shared<LinearModel>(modelOptions);

  // the order of the polynomial basis
  const std::size_t order = 2;

  // the number of points on a grid
  const std::size_t npoints = 5;

  pt::ptree suppOptions;
  suppOptions.put("NumNeighbors", npoints*npoints);
  suppOptions.put("BasisFunctions", "Basis1, Basis2");
  suppOptions.put("Basis1.Type", "TotalOrderPolynomials");
  suppOptions.put("Basis1.Order", order);
  suppOptions.put("Basis2.Type", "TotalOrderPolynomials");
  suppOptions.put("Basis2.Order", order);

  // add points on a grid
  std::vector<std::shared_ptr<SupportPoint> > supportPoints(4*npoints*npoints);
  for( std::size_t i=0; i<2*npoints; ++i ) {
    for( std::size_t j=0; j<2*npoints; ++j ) {
      supportPoints[2*npoints*i+j] = SupportPoint::Construct(
							     0.1*Eigen::Vector2d((double)i/(2*npoints-1)-0.5, (double)j/(2*npoints-1)-0.5),
							     model, suppOptions);
    }
  }

  // create the support point cloud
  pt::ptree ptSupportPointCloud;
  auto cloud = SupportPointCloud::Construct(supportPoints, ptSupportPointCloud);

  // the forcing function evaluated at each support point
  Eigen::MatrixXd forcing(2, supportPoints.size());
  for( std::size_t i=0; i<supportPoints.size(); ++i ) {
    forcing(0, i) = supportPoints[i]->x(0)*supportPoints[i]->x(1) + supportPoints[i]->x(1);
    forcing(1, i) = supportPoints[i]->x(1)*supportPoints[i]->x(1) + supportPoints[i]->x(0) + 2.0;
  }

  // create the local function
  pt::ptree ptFunc;
  auto func = std::make_shared<LocalFunctions>(cloud, ptFunc);

  pt::ptree optimizationOptions;
  optimizationOptions.put("Method", "NLopt");
  optimizationOptions.put("FunctionTolerance", 1.0e-4);
  optimizationOptions.put("NumThreads", 1);
  const double cost = func->ComputeOptimalCoefficients(forcing, optimizationOptions);

  for( std::size_t i=0; i<10; ++i ) {
    // pick a random point
    const Eigen::VectorXd x = 0.01*Eigen::VectorXd::Random(2);

    // evaluate the support point
    const Eigen::VectorXd eval = func->Evaluate(x);
    const Eigen::VectorXd expected = Eigen::Vector2d(x(0)*x(1)+x(1), x(1)*x(1)+x(0)+2.0);
    EXPECT_EQ(eval.size(), expected.size());
    EXPECT_NEAR((eval-expected).norm(), 0.0, 1.0e-14);
  }
}*/

/*TEST(LocalFunctionTests, CoupledLinearModel) {
  //EXPECT_TRUE(false);
}*/

/*
TEST_F(LocalFunctionsTests, CoupledComputation) {
  CreateSupportPointCloud<ExampleModelForLocalFunctionsTests>(0.5);

  // create the local function
  pt::ptree ptFunc;
  func = std::make_shared<LocalFunctions>(cloud, ptFunc);
}

class ExampleDifferentialModelForLocalFunctionsTests : public Model {
public:

  inline ExampleDifferentialModelForLocalFunctionsTests(pt::ptree const& pt) : Model(pt) {}

  virtual ~ExampleDifferentialModelForLocalFunctionsTests() = default;

protected:

  inline virtual Eigen::VectorXd OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    Eigen::VectorXd output = Eigen::VectorXd::Zero(outputDimension);
    for( std::size_t in=0; in<inputDimension; ++in ) { output(0) += FunctionDerivative(x, coefficients, bases, 0, in, 2); }

    assert(!std::isnan(output(0)));
    return output;
  }

  inline virtual double RightHandSideComponentImpl(Eigen::VectorXd const& x, std::size_t const outind) const override {
    assert(outind==0);
    return 6.0 + 2.0*x(0);
  }

private:
};

class ExampleBoundaryConditionsForLocalFunctionsTests : public Model {
public:

  inline ExampleBoundaryConditionsForLocalFunctionsTests(pt::ptree const& pt) : Model(pt) {}

  virtual ~ExampleBoundaryConditionsForLocalFunctionsTests() = default;

protected:

  inline virtual double RightHandSideComponentImpl(Eigen::VectorXd const& x, std::size_t const outind) const override {
    assert(outind==0);
    return 3.0*x(0)*x(0) + x(1) + x(0)*x(1)*x(1);
  }

private:
};

TEST(LocalFunctions_DifferentialOperatorsTests, PoissonEquation) {
  const std::size_t indim = 2, outdim = 1;
  pt::ptree modelOptions;
  modelOptions.put("InputDimension", indim);
  modelOptions.put("OutputDimension", outdim);

  // the order of the total order polynomial bases
  const std::size_t order = 3;

  // create a grid of support points
  const std::size_t npoints = 5;

  // options for the support point
  pt::ptree suppOptions;
  suppOptions.put("NumNeighbors", npoints*npoints);
  suppOptions.put("CoupledScale", 1.0);
  suppOptions.put("BasisFunctions", "Basis");
  suppOptions.put("Basis.Type", "TotalOrderPolynomials");
  suppOptions.put("Basis.Order", order);

  // create a support point cloud
  std::vector<std::shared_ptr<SupportPoint> > supportPoints(4*npoints*npoints);
  // add points on a grid so we know that they are well-poised
  for( std::size_t i=0; i<2*npoints; ++i ) {
    for( std::size_t j=0; j<2*npoints; ++j ) {
      if( i==0 || i==2*npoints-1 || j==0 || j==2*npoints-1 ) {
        supportPoints[2*npoints*i+j] = CoupledSupportPoint::Construct(
          0.1*Eigen::Vector2d((double)i/(2*npoints-1)-0.5, (double)j/(2*npoints-1)-0.5),
          std::make_shared<ExampleBoundaryConditionsForLocalFunctionsTests>(modelOptions),
          suppOptions);
      } else {
        supportPoints[2*npoints*i+j] = CoupledSupportPoint::Construct(
          0.1*Eigen::Vector2d((double)i/(2*npoints-1)-0.5, (double)j/(2*npoints-1)-0.5),
          std::make_shared<ExampleDifferentialModelForLocalFunctionsTests>(modelOptions),
          suppOptions);
      }
    }
  }

  // create the support point cloud
  pt::ptree ptSupportPointCloud;
  auto cloud = SupportPointCloud::Construct(supportPoints, ptSupportPointCloud);

  // create the local function
  pt::ptree ptFunc;
  auto func = std::make_shared<LocalFunctions>(cloud, ptFunc);

  // the cost of the optimial coefficients
  const double cost = func->CoefficientCost();
  EXPECT_NEAR(cost, 0.0, 1.0e-8);

  for( const auto& it : supportPoints ) {
    const Eigen::VectorXd eval = it->EvaluateLocalFunction(it->x);
    const Eigen::VectorXd expected = Eigen::VectorXd::Constant(1, 3.0*it->x(0)*it->x(0)+it->x(1)+it->x(0)*it->x(1)*it->x(1));
    EXPECT_EQ(eval.size(), expected.size());
    EXPECT_NEAR((eval-expected).norm(), 0.0, 10.0*std::sqrt(cost));
  }

  for( std::size_t i=0; i<10; ++i ) {
    // pick a random point
    const Eigen::VectorXd x = 0.01*Eigen::VectorXd::Random(indim);

    // find the nearest support point and the squared distance to it
    std::size_t ind; double dist;
    std::tie(ind, dist) = func->NearestNeighbor(x);
    for( const auto& it : supportPoints ) { EXPECT_TRUE(dist<=(x-it->x).dot(x-it->x)+1.0e-10); }

    // evaluate the support point
    const Eigen::VectorXd eval = func->Evaluate(x);
    const Eigen::VectorXd expected = Eigen::VectorXd::Constant(1, 3.0*x(0)*x(0)+x(1)+x(0)*x(1)*x(1));
    EXPECT_EQ(eval.size(), expected.size());
    EXPECT_NEAR((eval-expected).norm(), 0.0, 10.0*std::sqrt(cost));
  }
}
*/

} // namespace tests
} // namespace clf
