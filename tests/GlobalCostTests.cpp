#include <gtest/gtest.h>

#include "clf/LinearModel.hpp"
#include "clf/CoupledSupportPoint.hpp"
#include "clf/GlobalCost.hpp"

namespace pt = boost::property_tree;
using namespace clf;

class ExampleModelForGlobalCostTests : public LinearModel {
public:

  inline ExampleModelForGlobalCostTests(pt::ptree const& pt) : LinearModel(pt) {}

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
    suppOptions.put("BasisFunctions", "Basis1, Basis2");
    suppOptions.put("Basis1.Type", "TotalOrderSinCos");
    suppOptions.put("Basis1.Order", orderSinCos);
    suppOptions.put("Basis1.LocalBasis", false);
    suppOptions.put("Basis2.Type", "TotalOrderPolynomials");
    suppOptions.put("Basis2.Order", orderPoly);
    point = CoupledSupportPoint::Construct(
      Eigen::VectorXd::Random(indim),
      std::make_shared<ExampleModelForGlobalCostTests>(modelOptions),
      suppOptions);

    // create a support point cloud so that this point has nearest neighbors
    supportPoints.resize(4*npoints*npoints+1);
    supportPoints[0] = point;
    // add points on a grid so we know that they are well-poised---make sure there is an even number of points on each side so that the center point is not on the grid
    for( std::size_t i=0; i<2*npoints; ++i ) {
      for( std::size_t j=0; j<2*npoints; ++j ) {
        supportPoints[2*npoints*i+j+1] = CoupledSupportPoint::Construct(
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
private:
};

TEST_F(GlobalCostTests, CostEvaluationAndDerivatives) {
  pt::ptree pt;
  auto cost = std::make_shared<GlobalCost>(cloud, pt);
  EXPECT_EQ(cost->inputDimension, cloud->numCoefficients);
  std::size_t expectedValDim = 0;
  for( auto it=cloud->Begin(); it!=cloud->End(); ++it ) {
    auto point = std::dynamic_pointer_cast<SupportPoint>(*it);
    EXPECT_TRUE(point);

    expectedValDim += point->NumNeighbors()*point->model->outputDimension;
    for( std::size_t i=1; i<point->NumNeighbors(); ++i ) {
      const double coupling = point->CouplingFunction(i);
      if( coupling>1.0e-12 ) { expectedValDim += point->model->outputDimension; }
    }
  }
  EXPECT_EQ(cost->numPenaltyFunctions, expectedValDim);

  // choose random coefficients
  const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(cost->inputDimension);

  // compute the cost
  const Eigen::VectorXd costVec = cost->CostVector(coefficients);
  EXPECT_EQ(costVec.size(), cost->numPenaltyFunctions);

  // compute the exected cost
  Eigen::VectorXd expectedCost(cost->numPenaltyFunctions);
  {
    std::size_t ind = 0;
    for( std::size_t i=0; i<cloud->NumPoints(); ++i ) {
      auto point = cloud->GetSupportPoint(i);

      // compute the uncoupled cost
      expectedCost.segment(ind, cost->GetUncoupledCost(i)->numPenaltyFunctions) = cost->GetUncoupledCost(i)->CostVector(coefficients.segment(i*point->NumCoefficients(), point->NumCoefficients()));
      ind += cost->GetUncoupledCost(i)->numPenaltyFunctions;

      // compute the coupled cost
      for( const auto& coupled : cost->GetCoupledCost(i) ) {
        auto neigh = coupled->GetNeighbor();
        //expectedCost.segment(ind, coupled->numPenaltyFunctions) = coupled->ComputeCost(coefficients.segment(i*point->NumCoefficients(), point->NumCoefficients()), coefficients.segment(neigh->GlobalIndex()*point->NumCoefficients(), point->NumCoefficients()));
        ind += coupled->numPenaltyFunctions;
      }
    }
  }

  // check the cost
  EXPECT_EQ(costVec.size(), expectedCost.size());
  for( std::size_t i=0; i<costVec.size(); ++i ) { EXPECT_NEAR(costVec(i), expectedCost(i), 1.0e-12); }

  // compute the jacobian
  Eigen::SparseMatrix<double> jac;
  cost->Jacobian(coefficients, jac);

  Eigen::MatrixXd expectedJac = Eigen::MatrixXd::Zero(cost->numPenaltyFunctions, cost->inputDimension);
  {
    std::size_t ind = 0;
    for( std::size_t i=0; i<cloud->NumPoints(); ++i ) {
      auto point = cloud->GetSupportPoint(i);

      // compute the uncoupled cost entries
      std::vector<Eigen::Triplet<double> > localTriplets;
      //cost->GetUncoupledCost(i)->JacobianTriplets(coefficients.segment(i*point->NumCoefficients(), point->NumCoefficients()), localTriplets);
      for( const auto& it : localTriplets ) {
        const std::size_t row = ind+it.row();
        const std::size_t col = i*point->NumCoefficients()+it.col();
        expectedJac(row, col) = it.value();
      }
      localTriplets.clear();
      ind += cost->GetUncoupledCost(i)->numPenaltyFunctions;

      // compute the coupled cost entries
      for( const auto& coupled : cost->GetCoupledCost(i) ) {
        //coupled->JacobianTriplets(localTriplets);
        for( const auto& it : localTriplets ) {
          const std::size_t row = ind+it.row();
          const std::size_t col = (it.col()<point->NumCoefficients()? i : coupled->GetNeighbor()->GlobalIndex())*point->NumCoefficients() + it.col() - (it.col()<point->NumCoefficients()? 0 : point->NumCoefficients());
          expectedJac(row, col) = it.value();
        }
        localTriplets.clear();
        ind += coupled->numPenaltyFunctions;
      }
    }
  }

  // check the jacobian
  EXPECT_EQ(jac.rows(), expectedJac.rows());
  EXPECT_EQ(jac.cols(), expectedJac.cols());
  for( std::size_t i=0; i<jac.rows(); ++i ) { for( std::size_t j=0; j<jac.cols(); ++j ) { EXPECT_NEAR(jac.coeff(i, j), expectedJac(i, j), 1.0e-12); } }
}
