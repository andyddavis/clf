#include <gtest/gtest.h>

#include "clf/CoupledSupportPoint.hpp"
#include "clf/LocalFunctions.hpp"

namespace pt = boost::property_tree;
using namespace clf;

class ExampleModelForLocalFunctionsTests : public Model {
public:

  inline ExampleModelForLocalFunctionsTests(pt::ptree const& pt) : Model(pt) {}

  virtual ~ExampleModelForLocalFunctionsTests() = default;

protected:

  inline virtual double RightHandSideComponentImpl(Eigen::VectorXd const& x, std::size_t const outind) const override {
    if( outind==0 ) {
      // a sin/cos function
      return std::sin(M_PI*x(0))*std::cos(2.0*M_PI*x(1)) + std::cos(M_PI*x(1));
    } else if( outind==1 ) {
      // a quadratic function
      return x(1)*x(0) + x(0) + 1.0;
    } else {
      return 0.0;
    }
  }
private:
};

class LocalFunctionsTests : public::testing::Test {
protected:
  /// Set up information to test the support point cloud
  virtual void SetUp() override {}

  template<class MODEL>
  inline void CreateSupportPointCloud(double const couplingScale = 0.0) {
    pt::ptree modelOptions;
    modelOptions.put("InputDimension", indim);
    modelOptions.put("OutputDimension", outdim);

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

    // create a support point cloud
    supportPoints.resize(4*npoints*npoints);
    // add points on a grid so we know that they are well-poised
    for( std::size_t i=0; i<2*npoints; ++i ) {
      for( std::size_t j=0; j<2*npoints; ++j ) {
        supportPoints[2*npoints*i+j] = SupportPoint::Construct(
          0.1*Eigen::Vector2d((double)i/(2*npoints-1)-0.5, (double)j/(2*npoints-1)-0.5),
          std::make_shared<MODEL>(modelOptions),
          suppOptions);
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
    std::cout << "COMPUTED COST: " << cost << std::endl;
    EXPECT_NEAR(cost, 0.0, 1.0e-8);

    for( const auto& it : supportPoints ) {
      const Eigen::VectorXd eval = it->EvaluateLocalFunction(it->x);
      const Eigen::VectorXd expected = Eigen::Vector2d(std::sin(M_PI*it->x(0))*std::cos(2.0*M_PI*it->x(1)) + std::cos(M_PI*it->x(1)), it->x(1)*it->x(0) + it->x(0) + 1.0);
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
      const Eigen::VectorXd expected = Eigen::Vector2d(std::sin(M_PI*x(0))*std::cos(2.0*M_PI*x(1)) + std::cos(M_PI*x(1)), x(1)*x(0) + x(0) + 1.0);
      EXPECT_EQ(eval.size(), expected.size());
      EXPECT_NEAR((eval-expected).norm(), 0.0, 10.0*std::sqrt(cost));
    }
  }

  std::vector<std::shared_ptr<SupportPoint> > supportPoints;

  std::shared_ptr<SupportPointCloud> cloud;

  std::shared_ptr<LocalFunctions> func;

  /// The input and output dimensions
  const std::size_t indim = 2, outdim = 2;
};

TEST_F(LocalFunctionsTests, UncoupledComputation) {
  CreateSupportPointCloud<ExampleModelForLocalFunctionsTests>();

  // create the local function
  pt::ptree ptFunc;
  func = std::make_shared<LocalFunctions>(cloud, ptFunc);
}

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