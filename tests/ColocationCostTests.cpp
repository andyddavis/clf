#include <gtest/gtest.h>

#include <MUQ/Modeling/Distributions/RandomVariable.h>
#include <MUQ/Modeling/Distributions/Gaussian.h>

#include "clf/ColocationCost.hpp"
#include "clf/LevenbergMarquardt.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace clf;

class ExampleModelForColocationCostTests : public Model {
public:
  inline ExampleModelForColocationCostTests(pt::ptree const& pt) : Model(pt) {}

  virtual ~ExampleModelForColocationCostTests() = default;
private:

  /**
  @param[in] x The point \f$x \in \Omega \f$
  \return The evaluation of \f$f(x)\f$
  */
  inline virtual Eigen::VectorXd RightHandSideVectorImpl(Eigen::VectorXd const& x) const override { return Eigen::VectorXd::Constant(outputDimension, 1.0); }

  inline virtual Eigen::VectorXd OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    assert(bases.size()==outputDimension);
    Eigen::VectorXd output = IdentityOperator(x, coefficients, bases);
    output = output.array()*output.array();

    return output;
  }

  inline virtual Eigen::MatrixXd OperatorJacobianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(outputDimension, coefficients.size());

    std::size_t ind = 0;
    for( std::size_t i=0; i<outputDimension; ++i ) {
      std::size_t basisSize = bases[i]->NumBasisFunctions();
      const Eigen::VectorXd phi = bases[i]->EvaluateBasisFunctions(x);
      assert(phi.size()==basisSize);
      jac.row(i).segment(ind, basisSize) = phi.dot(coefficients.segment(ind, basisSize))*phi;
      ind += basisSize;
    }

    return 2.0*jac;
  }
};

class ColocationCostTests : public::testing::Test {
protected:
  virtual void SetUp() override {
    pt::ptree modelOptions;
    modelOptions.put("InputDimension", indim);
    modelOptions.put("OutputDimension", outdim);
    model = std::make_shared<ExampleModelForColocationCostTests>(modelOptions);

    pt::ptree ptSupportPoints;
    ptSupportPoints.put("BasisFunctions", "Basis1, Basis2");
    ptSupportPoints.put("Basis1.Type", "TotalOrderPolynomials");
    ptSupportPoints.put("Basis2.Type", "TotalOrderPolynomials");

    // the number of support points
    const std::size_t n = 100;

    // create a bunch of random support points
    std::vector<std::shared_ptr<SupportPoint> > supportPoints(n);
    auto dist = std::make_shared<Gaussian>(indim)->AsVariable();
    for( std::size_t i=0; i<n; ++i ) { supportPoints[i] = SupportPoint::Construct(dist->Sample(), model, ptSupportPoints); }

    // create the support point cloud
    pt::ptree ptSupportPointCloud;
    supportCloud = SupportPointCloud::Construct(supportPoints, ptSupportPointCloud);

    // the distribution we sample the colocation points from
    sampler = std::make_shared<ColocationPointSampler>(dist, model);
  }

  virtual void TearDown() override {
    EXPECT_EQ(cost->inputDimension, model->outputDimension*supportCloud->NumPoints());
  }

  /// Options for the colocation point cloud function
  pt::ptree cloudOptions;

  /// The domain dimension
  const std::size_t indim = 3;

  /// The output dimension
  const std::size_t outdim = 2;

  // The model (default to just using the identity)
  std::shared_ptr<Model> model;

  /// The support point cloud
  std::shared_ptr<SupportPointCloud> supportCloud;

  /// The colocation cost
  std::shared_ptr<ColocationCost> cost;

  /// The distribution we sample the colocation points from
  std::shared_ptr<ColocationPointSampler> sampler;
};

TEST_F(ColocationCostTests, Construction) {
  // the number of colocation points
  const std::size_t nColocPoints = 125;

  // options for the cost function
  cloudOptions.put("NumColocationPoints", nColocPoints);
  auto colocationCloud = std::make_shared<ColocationPointCloud>(sampler, supportCloud, cloudOptions);

  // create the colocation cost
  cost = std::make_shared<ColocationCost>(colocationCloud);
  EXPECT_EQ(cost->numPenaltyFunctions, model->outputDimension*nColocPoints);
}

TEST_F(ColocationCostTests, ComputeOptimalCoefficients) {
  auto colocationCloud = std::make_shared<ColocationPointCloud>(sampler, supportCloud, cloudOptions);

  // create the colocation cost
  cost = std::make_shared<ColocationCost>(colocationCloud);
  EXPECT_EQ(cost->numPenaltyFunctions, model->outputDimension*supportCloud->NumPoints());

  Eigen::MatrixXd data(outdim, supportCloud->NumPoints());
  for( std::size_t i=0; i<data.cols(); ++i ) { data.col(i) = supportCloud->GetSupportPoint(i)->x.head(outdim); }

  cost->ComputeOptimalCoefficients(data);

  for( auto it=supportCloud->Begin(); it!=supportCloud->End(); ++it ) {
    auto point = std::dynamic_pointer_cast<SupportPoint>(*it);
    EXPECT_TRUE(point);

    const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);
    EXPECT_NEAR((x.head(outdim)-point->EvaluateLocalFunction(x)).norm(), 0.0, 1.0e-8);
  }
}

TEST_F(ColocationCostTests, CostFunctionEvaluation) {
  // the number of colocation points
  const std::size_t nColocPoints = 25;

  // options for the cost function
  cloudOptions.put("NumColocationPoints", nColocPoints);

  auto colocationCloud = std::make_shared<ColocationPointCloud>(sampler, supportCloud, cloudOptions);

  // sample the colocation points
  colocationCloud->Resample();

  // create the colocation cost
  cost = std::make_shared<ColocationCost>(colocationCloud);
  EXPECT_EQ(cost->numPenaltyFunctions, model->outputDimension*nColocPoints);

  Eigen::MatrixXd data(outdim, supportCloud->NumPoints());
  for( std::size_t i=0; i<data.cols(); ++i ) { data.col(i) = supportCloud->GetSupportPoint(i)->x.head(outdim); }

  const Eigen::VectorXd costEval = cost->CostVector(Eigen::Map<const Eigen::VectorXd>(data.data(), data.size()));

  EXPECT_EQ(costEval.size(), cost->numPenaltyFunctions);
  for( std::size_t i=0; i<colocationCloud->numColocationPoints; ++i ) {
    auto it = colocationCloud->GetColocationPoint(i);
    EXPECT_TRUE(it);
    EXPECT_NEAR((it->Operator() - it->RightHandSide() - costEval.segment(i*model->outputDimension, model->outputDimension)).norm(), 0.0, 1.0e-10);
  }

  Eigen::MatrixXd jacFD(cost->numPenaltyFunctions, cost->inputDimension);

  Eigen::SparseMatrix<double> jac;
  cost->Jacobian(Eigen::Map<const Eigen::VectorXd>(data.data(), data.size()), jac);
  {
    const double dc = 1.0e-8;
    Eigen::Map<const Eigen::VectorXd> dataVec(data.data(), data.size());
    for( std::size_t c=0; c<data.size(); ++c ) {
      Eigen::VectorXd datap = dataVec;
      datap(c) += dc;
      Eigen::VectorXd datam = dataVec;
      datam(c) -= dc;
      Eigen::VectorXd data2m = dataVec;
      data2m(c) -= 2.0*dc;

      const Eigen::VectorXd costp = cost->CostVector(datap);
      const Eigen::VectorXd costm = cost->CostVector(datam);
      const Eigen::VectorXd cost2m = cost->CostVector(data2m);

      jacFD.col(c) = (2.0*costp + 3.0*costEval - 6.0*costm + cost2m)/(6.0*dc);
    }
  }

  EXPECT_EQ(jac.rows(), jacFD.rows());
  EXPECT_EQ(jac.cols(), jacFD.cols());
  for( std::size_t i=0; i<jac.rows(); ++i ) {
    for( std::size_t j=0; j<jac.cols(); ++j ) {
      EXPECT_NEAR(jac.coeff(i, j), jacFD(i, j), 1.0e-4);
    }
  }
}

TEST_F(ColocationCostTests, CostFunctionMinimization) {
  // the number of colocation points
  const std::size_t nColocPoints = 100000;

  // options for the cost function
  cloudOptions.put("NumColocationPoints", nColocPoints);

  auto colocationCloud = std::make_shared<ColocationPointCloud>(sampler, supportCloud, cloudOptions);

  // sample the colocation points
  colocationCloud->Resample();

  // create the colocation cost
  cost = std::make_shared<ColocationCost>(colocationCloud);
  EXPECT_EQ(cost->numPenaltyFunctions, model->outputDimension*nColocPoints);

  pt::ptree pt;
  auto lm = std::make_shared<SparseLevenbergMarquardt>(cost, pt);
  Eigen::VectorXd costVec;
  Eigen::VectorXd data(outdim*supportCloud->NumPoints());
  for( std::size_t i=0; i<supportCloud->NumPoints(); ++i ) {

    data.segment(i*outdim, outdim) = Eigen::VectorXd::Constant(outdim, supportCloud->GetSupportPoint(i)->x(1));
  }
  //std::cout << "TEST: " << cost->CostVector(data).transpose() << std::endl;
  lm->Minimize(data, costVec);
  auto dist = std::make_shared<Gaussian>(indim)->AsVariable();
  //for( std::size_t i=0; i<supportCloud->NumPoints(); ++i ) {
  for( std::size_t i=0; i<std::min((std::size_t)10, colocationCloud->numColocationPoints); ++i ) {
    //const Eigen::VectorXd pnt = dist->Sample();
    //const Eigen::VectorXd pnt = supportCloud->GetSupportPoint(i)->x;
    const Eigen::VectorXd pnt = colocationCloud->GetColocationPoint(i)->x;
    auto support = colocationCloud->GetColocationPoint(i)->supportPoint.lock();
    std::cout << "x: " << pnt.transpose() << std::endl;
    std::cout << "||x||: " << pnt.norm() << std::endl;
    std::cout << "x.x: " << pnt.dot(pnt) << std::endl;
    std::cout << "f(x): " << support->RightHandSide(pnt).transpose() << std::endl;
    std::cout << "u(x): " << support->EvaluateLocalFunction(pnt).transpose() << std::endl;
    std::cout << "L(u(x)): " << support->Operator(pnt).transpose() << std::endl;
    std::cout << std::endl;
  }
}
