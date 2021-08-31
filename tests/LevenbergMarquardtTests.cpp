#include <gtest/gtest.h>

#include <Eigen/Sparse>

#include "clf/LevenbergMarquardt.hpp"

namespace pt = boost::property_tree;
using namespace clf;

class DenseCostTest : public DenseCostFunction {
public:

  inline DenseCostTest() : DenseCostFunction(3, 4) {}

  virtual ~DenseCostTest() = default;

protected:

  inline virtual Eigen::VectorXd CostImpl(Eigen::VectorXd const& beta) const override {
    Eigen::VectorXd cost(4);

    cost(0) = beta(0);
    cost(1) = beta(1)-1.0;
    cost(2) = beta(2);
    cost(3) = beta(0)*beta(2);

    return cost;
  }

  inline virtual void JacobianImpl(Eigen::VectorXd const& beta, Eigen::MatrixXd& jac) const override {
    jac(0, 0) = 1.0;
    jac(1, 1) = 1.0;
    jac(2, 2) = 1.0;
    jac(3, 0) = beta(2); jac(3, 2) = beta(0);
  }

private:
};

TEST(LevenbergMarquardtTests, Dense) {
  auto cost = std::make_shared<DenseCostTest>();
  EXPECT_EQ(cost->inDim, 3);
  EXPECT_EQ(cost->valDim, 4);

  pt::ptree pt;
  auto lm = std::make_shared<DenseLevenbergMarquardt>(cost, pt);
  EXPECT_EQ(lm->maxEvals, 1000);
  EXPECT_EQ(lm->maxJacEvals, 1000);
  EXPECT_EQ(lm->maxIters, 1000);
  EXPECT_NEAR(lm->gradTol, 1.0e-8, 1.0e-14);
  EXPECT_NEAR(lm->funcTol, 1.0e-8, 1.0e-14);
  EXPECT_NEAR(lm->betaTol, 1.0e-8, 1.0e-14);

  Eigen::VectorXd beta = Eigen::VectorXd::Random(cost->inDim);
  Eigen::VectorXd costVec;
  lm->Minimize(beta, costVec);
  const double totCost = costVec.dot(costVec);
  EXPECT_NEAR(totCost, 0.0, 1.0e-10);

  EXPECT_NEAR(beta(0), 0.0, 2.0*std::sqrt(totCost));
  EXPECT_NEAR(beta(1), 1.0, 2.0*std::sqrt(totCost));
  EXPECT_NEAR(beta(2), 0.0, 2.0*std::sqrt(totCost));
}

class SparseCostTest : public SparseCostFunction {
public:

  inline SparseCostTest() : SparseCostFunction(3, 4) {}

  virtual ~SparseCostTest() = default;

protected:

  inline virtual Eigen::VectorXd CostImpl(Eigen::VectorXd const& beta) const override {
    Eigen::VectorXd cost(4);
    cost(0) = beta(0);
    cost(1) = beta(1)-1.0;
    cost(2) = beta(2);
    cost(3) = beta(0)*beta(2);

    return cost;
  }

  inline virtual void JacobianImpl(Eigen::VectorXd const& beta, Eigen::SparseMatrix<double>& jac) const override {
    jac.coeffRef(0, 0) = 1.0;
    jac.coeffRef(1, 1) = 1.0;
    jac.coeffRef(2, 2) = 1.0;
    jac.coeffRef(3, 0) = beta(2); jac.coeffRef(3, 2) = beta(0);
  }

private:
};

TEST(LevenbergMarquardtTests, Sparse) {
  auto cost = std::make_shared<SparseCostTest>();
  EXPECT_EQ(cost->inDim, 3);
  EXPECT_EQ(cost->valDim, 4);

  pt::ptree pt;

  auto lm = std::make_shared<SparseLevenbergMarquardt>(cost, pt);
  EXPECT_EQ(lm->maxEvals, 1000);
  EXPECT_EQ(lm->maxJacEvals, 1000);
  EXPECT_EQ(lm->maxIters, 1000);
  EXPECT_NEAR(lm->gradTol, 1.0e-8, 1.0e-14);
  EXPECT_NEAR(lm->funcTol, 1.0e-8, 1.0e-14);
  EXPECT_NEAR(lm->betaTol, 1.0e-8, 1.0e-14);

  Eigen::VectorXd beta = Eigen::VectorXd::Random(cost->inDim);
  Eigen::VectorXd costVec;
  lm->Minimize(beta, costVec);
  const double totCost = costVec.sum();
  EXPECT_NEAR(totCost, 0.0, 1.0e-10);

  EXPECT_NEAR(beta(0), 0.0, 2.0*std::sqrt(totCost));
  EXPECT_NEAR(beta(1), 1.0, 2.0*std::sqrt(totCost));
  EXPECT_NEAR(beta(2), 0.0, 2.0*std::sqrt(totCost));
}
