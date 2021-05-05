#include <gtest/gtest.h>

#include "clf/CoupledCost.hpp"
#include "clf/SupportPointCloud.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
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
  virtual void SetUp() override {
    pt::ptree modelOptions;
    modelOptions.put("InputDimension", indim);
    modelOptions.put("OutputDimension", outdim);

    // the order of the total order polynomial and sin/cos bases
    const std::size_t orderPoly = 4, orderSinCos = 2;

    const std::size_t npoints = 6;

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
  // create the uncoupled cost for each neighbor
  for( auto it=supportPoints.begin(); it!=supportPoints.end(); ++it ) {
    pt::ptree costOptions;
    costOptions.put("CoupledScale", coupledScale);
    auto cost = std::make_shared<CoupledCost>(point, *it, costOptions);
    EXPECT_EQ(cost->inputSizes(0), point->NumCoefficients()+(*it)->NumCoefficients());
    EXPECT_DOUBLE_EQ((*it)->couplingScale, coupledScale);
    EXPECT_EQ(cost->Coupled(), (*it)!=point & point->IsNeighbor((*it)->GlobalIndex()));

    // choose random coefficients
    const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(cost->inputSizes(0));

    // compute the coupling cost
    const double cst = cost->CostFunction::Cost(ref_vector<Eigen::VectorXd>(1, coefficients));
    if( !cost->Coupled() ) {
      EXPECT_DOUBLE_EQ(cst, 0.0);
    } else {
      const Eigen::VectorXd pntEval = point->EvaluateLocalFunction((*it)->x, coefficients.head(point->NumCoefficients()));
      const Eigen::VectorXd neighEval = (*it)->EvaluateLocalFunction((*it)->x, coefficients.tail((*it)->NumCoefficients()));
      const Eigen::VectorXd diff = pntEval - neighEval;
      EXPECT_NEAR(cst, coupledScale*diff.dot(diff)*point->NearestNeighborKernel(point->LocalIndex((*it)->GlobalIndex()))/2.0, 1.0e-10);
    }

    // compute the coupling gradient
    const Eigen::VectorXd grad = cost->CostFunction::Gradient(0, std::vector<Eigen::VectorXd>(1, coefficients), (0.75*Eigen::VectorXd::Ones(1)).eval());
    EXPECT_EQ(grad.size(), coefficients.size());
    if( !cost->Coupled() ) {
      EXPECT_DOUBLE_EQ(grad.norm(), 0.0);
    } else {
      const Eigen::VectorXd gradFD = cost->GradientByFD(0, 0, ref_vector<Eigen::VectorXd>(1, coefficients), 0.75*Eigen::VectorXd::Ones(1));
      EXPECT_NEAR((gradFD-grad).norm()/gradFD.norm(), 0.0, 1.0e-6);
    }

    // comptue the coupling hessian
    if( !cost->Coupled() ) {
      std::vector<Eigen::MatrixXd> ViVi, ViVj, VjVj;
      cost->Hessian(ViVi, ViVj, VjVj);
      EXPECT_EQ(ViVi.size(), 0); EXPECT_EQ(ViVj.size(), 0); EXPECT_EQ(VjVj.size(), 0);
    } else {
      const Eigen::MatrixXd hessFD = cost->HessianByFD(0, std::vector<Eigen::VectorXd>(1, coefficients));
      EXPECT_EQ(hessFD.rows(), point->NumCoefficients()+(*it)->NumCoefficients());
      EXPECT_EQ(hessFD.cols(), point->NumCoefficients()+(*it)->NumCoefficients());

      Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(hessFD.rows(), hessFD.cols());

      std::vector<Eigen::MatrixXd> ViVi, ViVj, VjVj;
      cost->Hessian(ViVi, ViVj, VjVj);
      EXPECT_EQ(ViVi.size(), outdim);
      EXPECT_EQ(ViVj.size(), outdim);
      EXPECT_EQ(VjVj.size(), outdim);

      std::size_t rows = 0, cols = 0, ind = 0, jnd = 0;
      for( const auto& V : ViVi ) {
        rows += V.rows(); cols += V.cols();
        hess.block(ind, jnd, V.rows(), V.cols()) = V;
        ind += V.rows(); jnd += V.cols();
      }
      EXPECT_EQ(rows, point->NumCoefficients());
      EXPECT_EQ(cols, point->NumCoefficients());

      rows = 0; cols = 0;
      ind = 0; jnd = point->NumCoefficients();
      for( const auto& V : ViVj ) {
        rows += V.rows(); cols += V.cols();
        hess.block(ind, jnd, V.rows(), V.cols()) = V;
        hess.block(jnd, ind, V.cols(), V.rows()) = V.transpose();
        ind += V.rows(); jnd += V.cols();
      }
      EXPECT_EQ(rows, point->NumCoefficients());
      EXPECT_EQ(cols, (*it)->NumCoefficients());

      rows = 0; cols = 0;
      ind = point->NumCoefficients(); jnd = point->NumCoefficients();
      for( const auto& V : VjVj ) {
        rows += V.rows(); cols += V.cols();
        hess.block(ind, jnd, V.rows(), V.cols()) = V;
        ind += V.rows(); jnd += V.cols();
      }
      EXPECT_EQ(rows, (*it)->NumCoefficients());
      EXPECT_EQ(cols, (*it)->NumCoefficients());

      EXPECT_EQ(hess.rows(), point->NumCoefficients()+(*it)->NumCoefficients());
      EXPECT_EQ(hess.cols(), point->NumCoefficients()+(*it)->NumCoefficients());

      EXPECT_NEAR((hess-hessFD).norm(), 0.0, 1.0e-6);
    }
  }
}
