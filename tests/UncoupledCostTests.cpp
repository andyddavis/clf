#include <gtest/gtest.h>

#include "clf/UncoupledCost.hpp"
#include "clf/SupportPointCloud.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
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

  inline virtual Eigen::VectorXd OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    assert(bases.size()==outputDimension);
    Eigen::VectorXd output = IdentityOperator(x, coefficients, bases);
    output += (output.array()*output.array()).matrix();

    return output;
  }

  inline virtual Eigen::MatrixXd OperatorJacobianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(outputDimension, coefficients.size());

    std::size_t ind = 0;
    for( std::size_t i=0; i<outputDimension; ++i ) {
      std::size_t basisSize = bases[i]->NumBasisFunctions();
      const Eigen::VectorXd phi = bases[i]->EvaluateBasisFunctions(x);
      assert(phi.size()==basisSize);

      jac.row(i).segment(ind, basisSize) = 2.0*phi.dot(coefficients.segment(ind, basisSize))*phi + phi;

      ind += basisSize;
    }

    return jac;
  }

  /// Compute the true Hessian of the operator with respect to the coefficients
  inline std::vector<Eigen::MatrixXd> OperatorHessianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
    std::vector<Eigen::MatrixXd> hess(outputDimension, Eigen::MatrixXd::Zero(coefficients.size(), coefficients.size()));

    std::size_t ind = 0;
    for( std::size_t i=0; i<outputDimension; ++i ) {
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
  virtual void SetUp() override {
    // the input and output dimensions
    const std::size_t indim = 3, outdim = 2;

    pt::ptree modelOptions;
    modelOptions.put("InputDimension", indim);
    modelOptions.put("OutputDimension", outdim);

    // the order of the total order polynomial and sin/cos bases
    const std::size_t orderPoly = 5, orderSinCos = 2;

    // options for the support point
    pt::ptree suppOptions;
    suppOptions.put("NumNeighbors", 75);
    suppOptions.put("BasisFunctions", "Basis1, Basis2");
    suppOptions.put("Basis1.Type", "TotalOrderSinCos");
    suppOptions.put("Basis1.Order", orderSinCos);
    suppOptions.put("Basis2.Type", "TotalOrderPolynomials");
    suppOptions.put("Basis2.Order", orderPoly);
    point = SupportPoint::Construct(
      Eigen::VectorXd::Random(indim),
      std::make_shared<ExampleIdentityModelForUncoupledCostTests>(modelOptions),
      suppOptions);

    // create a support point cloud so that this point has nearest neighbors
    supportPoints.resize(75);
    supportPoints[0] = point;
    for( std::size_t i=1; i<supportPoints.size(); ++i ) {
      supportPoints[i] = SupportPoint::Construct(
        point->x + 0.1*Eigen::VectorXd::Random(indim), // choose a bunch of points that are close to the point we care about
        std::make_shared<ExampleIdentityModelForUncoupledCostTests>(modelOptions),
        suppOptions);
      }
      pt::ptree ptSupportPointCloud;
      cloud = SupportPointCloud::Construct(supportPoints, ptSupportPointCloud);
    }

  /// Make sure everything is what we expect
  virtual void TearDown() override {}

  std::vector<std::shared_ptr<SupportPoint> > supportPoints;

  std::shared_ptr<SupportPoint> point;

  std::shared_ptr<SupportPointCloud> cloud;

  const double regularizationScale = 0.5;
private:
};

TEST_F(UncoupledCostTests, CostEvaluationAndDerivatives) {
  // create the uncoupled cost
  pt::ptree costOptions;
  costOptions.put("RegularizationParameter", regularizationScale);
  auto cost = std::make_shared<UncoupledCost>(point, costOptions);
  EXPECT_EQ(cost->inputSizes(0), point->NumCoefficients());
  EXPECT_DOUBLE_EQ(cost->regularizationScale, regularizationScale);

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
    trueCost += regularizationScale*coefficients.dot(coefficients);
    trueCost /= (2.0*supportPoints.size());
  }

  // compute the cost
  const double cst = cost->Cost(coefficients);
  EXPECT_NEAR(cst, trueCost, 1.0e-10);

  // compute the gradient
  const Eigen::VectorXd gradFD = cost->GradientByFD(0, 0, ref_vector<Eigen::VectorXd>(1, coefficients), 0.75*Eigen::VectorXd::Ones(1));
  EXPECT_EQ(gradFD.size(), coefficients.size());
  const Eigen::VectorXd grad = cost->Gradient(0, std::vector<Eigen::VectorXd>(1, coefficients), (0.75*Eigen::VectorXd::Ones(1)).eval());
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

TEST_F(UncoupledCostTests, MinimizeOnePoint) {
  point->MinimizeUncoupledCost();

  for( const auto& it : supportPoints ) {
    const Eigen::VectorXd eval = point->EvaluateLocalFunction(it->x);
    const Eigen::VectorXd operatorEval = (eval.array()*eval.array()).matrix()+eval;
    EXPECT_EQ(eval.size(), point->model->outputDimension);
    std::cout << it->x.prod() << std::endl;
    std::cout << operatorEval.transpose() << std::endl;
    std::cout << std::endl;
    //for( std::size_t i=0; i<eval.size(); ++i ) { EXPECT_NEAR(it->x.prod(), operatorEval(i), 1.0e-12); }
  }
}
