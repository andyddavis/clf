#include <gtest/gtest.h>

#include "clf/UncoupledCost.hpp"
#include "clf/SupportPointCloud.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::Optimization;
using namespace clf;

class ExampleModelForUncoupledCostTests : public Model {
public:

  inline ExampleModelForUncoupledCostTests(pt::ptree const& pt) : Model(pt) {}

  virtual ~ExampleModelForUncoupledCostTests() = default;

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

  inline virtual Eigen::VectorXd OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    assert(bases.size()==outputDimension);
    Eigen::VectorXd output = IdentityOperator(x, coefficients, bases);
    output(1) += output(1)*output(1);

    return output;
  }

  inline virtual Eigen::MatrixXd OperatorJacobianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(outputDimension, coefficients.size());

    std::size_t ind = 0;
    for( std::size_t i=0; i<outputDimension; ++i ) {
      std::size_t basisSize = bases[i]->NumBasisFunctions();
      const Eigen::VectorXd phi = bases[i]->EvaluateBasisFunctions(x);
      assert(phi.size()==basisSize);

      if( i==1 ) {
        jac.row(i).segment(ind, basisSize) = phi + 2.0*phi.dot(coefficients.segment(ind, basisSize))*phi;
      } else {
        jac.row(i).segment(ind, basisSize) = phi;
      }

      ind += basisSize;
    }

    return jac;
  }

  /// Compute the true Hessian of the operator with respect to the coefficients
  inline std::vector<Eigen::MatrixXd> OperatorHessianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
    std::vector<Eigen::MatrixXd> hess(outputDimension, Eigen::MatrixXd::Zero(coefficients.size(), coefficients.size()));

    std::size_t ind = bases[0]->NumBasisFunctions();
    for( std::size_t i=1; i<outputDimension; ++i ) {
      Eigen::VectorXd phi = bases[i]->EvaluateBasisFunctions(x);
      for( std::size_t k=0; k<phi.size(); ++k ) {
        for( std::size_t j=0; j<phi.size(); ++j ) {
          hess[i](ind+k, ind+j) = 2.0*phi(k)*phi(j);
        }
      }
      ind += bases[i]->NumBasisFunctions();
    }

    return hess;
  }

private:
};

class UncoupledCostTests : public::testing::Test {
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

    // must be odd so that the center is not a point on the grid
    const std::size_t npoints = 7;

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
      Eigen::VectorXd::Ones(indim),
      std::make_shared<ExampleModelForUncoupledCostTests>(modelOptions),
      suppOptions);

    // create a support point cloud so that this point has nearest neighbors
    supportPoints.resize(suppOptions.get<std::size_t>("NumNeighbors"));
    supportPoints[0] = point;
    // add points on a grid so we know that they are well-poised
    for( std::size_t i=0; i<npoints; ++i ) {
      for( std::size_t j=0; j<npoints; ++j ) {
        supportPoints[i*npoints+j+1] = SupportPoint::Construct(
          point->x+0.1*Eigen::Vector2d((double)i/npoints-0.5, (double)j/npoints-0.5),
          std::make_shared<ExampleModelForUncoupledCostTests>(modelOptions),
          suppOptions);
      }
    }
    pt::ptree ptSupportPointCloud;
    cloud = SupportPointCloud::Construct(supportPoints, ptSupportPointCloud);
  }

  void CheckMinimization(double const costtol) const {
    const double cost = point->MinimizeUncoupledCost();
    const double tol = 10.0*std::sqrt(cost);
    EXPECT_TRUE(cost<costtol);

    for( const auto& it : supportPoints ) {
      const Eigen::VectorXd eval = point->EvaluateLocalFunction(it->x);
      const Eigen::VectorXd operatorEval = Eigen::Vector2d(eval(0), eval(1)+eval(1)*eval(1));
      const Eigen::VectorXd expectedRHS = Eigen::Vector2d(
        std::sin(2.0*M_PI*it->x(0))*std::cos(M_PI*it->x(1)) + std::cos(it->x(0)),
        it->x.prod()
      );
      EXPECT_EQ(eval.size(), point->model->outputDimension);
      for( std::size_t i=0; i<eval.size(); ++i ) {
        EXPECT_NEAR(expectedRHS(i), operatorEval(i), tol);
      }
    }
  }

  /// Make sure everything is what we expect
  virtual void TearDown() override {}

  /// The input and output dimensions
  const std::size_t indim = 2, outdim = 2;

  std::vector<std::shared_ptr<SupportPoint> > supportPoints;

  std::shared_ptr<SupportPoint> point;

  std::shared_ptr<SupportPointCloud> cloud;

  const double regularizationScale = 0.5;

  const double uncoupledScale = 0.25;
private:
};

TEST_F(UncoupledCostTests, CostEvaluationAndDerivatives) {
  CreateCloud();

  // create the uncoupled cost
  pt::ptree costOptions;
  costOptions.put("RegularizationParameter", regularizationScale);
  costOptions.put("UncoupledScale", uncoupledScale);
  auto cost = std::make_shared<UncoupledCost>(point, costOptions);
  EXPECT_EQ(cost->inputSizes(0), point->NumCoefficients());
  EXPECT_DOUBLE_EQ(cost->regularizationScale, regularizationScale);
  EXPECT_DOUBLE_EQ(cost->uncoupledScale, uncoupledScale);

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
      const Eigen::VectorXd diff = point->Operator(supportPoints[point->GlobalNeighborIndex(i)]->x, coefficients) - point->model->RightHandSide(supportPoints[point->GlobalNeighborIndex(i)]->x);

      trueCost += uncoupledScale*kernel(i)*diff.dot(diff);
    }
    trueCost += regularizationScale*coefficients.dot(coefficients);
    trueCost /= (2.0*supportPoints.size());
  }

  // compute the cost
  const double cst = cost->Cost(coefficients);
  EXPECT_NEAR(cst, trueCost, 1.0e-10);

  // compute the gradient
  const Eigen::VectorXd gradFD = cost->GradientByFD(0, 0, ref_vector<Eigen::VectorXd>(1, coefficients), 0.75*Eigen::VectorXd::Ones(1));
  EXPECT_EQ(gradFD.size(), coefficients.size());
  const Eigen::VectorXd grad = cost->CostFunction::Gradient(0, std::vector<Eigen::VectorXd>(1, coefficients), (0.75*Eigen::VectorXd::Ones(1)).eval());
  EXPECT_EQ(grad.size(), coefficients.size());
  EXPECT_NEAR((gradFD-grad).norm()/gradFD.norm(), 0.0, 1.0e-5);

  // compute the Hessian
  const Eigen::MatrixXd hessFD = cost->HessianByFD(0, std::vector<Eigen::VectorXd>(1, coefficients));
  EXPECT_EQ(hessFD.rows(), coefficients.size());
  EXPECT_EQ(hessFD.cols(), coefficients.size());
  const Eigen::MatrixXd hessGN = cost->Hessian(coefficients, true);
  EXPECT_EQ(hessGN.rows(), coefficients.size());
  EXPECT_EQ(hessGN.cols(), coefficients.size());
  const Eigen::MatrixXd hess = cost->Hessian(coefficients, false);
  EXPECT_EQ(hess.rows(), coefficients.size());
  EXPECT_EQ(hess.cols(), coefficients.size());

  // the finite difference and the exact Hessian should be similar
  EXPECT_NEAR((hessFD-hess).norm()/hessFD.norm(), 0.0, 1.0e-5);
}

TEST_F(UncoupledCostTests, MinimizeOnePointNLOPT) {
  pt::ptree optimization;
  optimization.put("UseNLOPT", true);
  optimization.put("AbsoluteFunctionTol", 0.0);
  optimization.put("RelativeFunctionTol", 0.0);
  optimization.put("AbsoluteStepSizeTol", 0.0);
  optimization.put("RelativeStepSizeTol", 0.0);
  optimization.put("MaxEvaluations", 100000);
  CreateCloud(optimization);

  CheckMinimization(1.0e-8);
}

TEST_F(UncoupledCostTests, MinimizeOnePointNewtonsMethod_GaussNewtonHessian) {
  pt::ptree optimization;
  optimization.put("UseNLOPT", false);
  optimization.put("UseGaussNewtonHessian", true);
  CreateCloud(optimization);

  CheckMinimization(1.0e-8);
}

TEST_F(UncoupledCostTests, MinimizeOnePointNewtonsMethod_TrueHessian) {
  pt::ptree optimization;
  optimization.put("UseNLOPT", false);
  optimization.put("UseGaussNewtonHessian", false);
  CreateCloud(optimization);

  CheckMinimization(1.0e-8);
}
