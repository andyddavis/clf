#include <gtest/gtest.h>

#include "clf/LinearModel.hpp"
#include "clf/PolynomialBasis.hpp"

namespace pt = boost::property_tree;

namespace clf {
namespace tests {

/// A class that runs the tests for clf::LinearModel
class LinearModelTests : public::testing::Test {
protected: 
  /// Check the linear model
  virtual void TearDown() override {
    EXPECT_TRUE(model->IsLinear());
    
    std::vector<std::shared_ptr<const BasisFunctions> > bases(outdim);
    
    std::size_t ncoeffs = 0;
    for( std::size_t i=0; i<outdim; ++i ) {
      pt::ptree polyBasisOptions;
      polyBasisOptions.put("InputDimension", indim);
      polyBasisOptions.put("Order", 2);
      bases[i] = PolynomialBasis::TotalOrderBasis(polyBasisOptions);
      ncoeffs += bases[i]->NumBasisFunctions();
    }
    
    // pick a random point
    const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);
    
    // the coefficients
    const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(ncoeffs);
    
    const Eigen::VectorXd expected = model->IdentityOperator(x, coefficients, bases);
    const Eigen::VectorXd ux = model->Operator(x, coefficients, bases);
    EXPECT_NEAR((expected-ux).norm(), 0.0, 1.0e-14);
    
    const Eigen::MatrixXd expectedJac = model->IdentityOperator(x, coefficients, bases);
    const Eigen::MatrixXd jac = model->Operator(x, coefficients, bases);
    EXPECT_NEAR((expectedJac-jac).norm(), 0.0, 1.0e-14);
    
    const std::vector<Eigen::MatrixXd> hess = model->OperatorHessian(x, coefficients, bases);
    EXPECT_EQ(hess.size(), outdim);
    for( const auto& it : hess ) {
      EXPECT_EQ(it.cols(), ncoeffs);
      EXPECT_EQ(it.rows(), ncoeffs);
      EXPECT_NEAR(it.norm(), 0.0, 1.0e-14);
    }
  }

  /// The input dimension
  const std::size_t indim = 3;

  /// The output dimension
  const std::size_t outdim = 2;

  /// The linear model 
  std::shared_ptr<LinearModel> model;
};

TEST_F(LinearModelTests, IdentityFromPtree) {
  pt::ptree options;
  options.put("InputDimension", indim);
  options.put("OutputDimension", outdim);
  model = std::make_shared<LinearModel>(options);
}

TEST_F(LinearModelTests, IdentityFromInOutDim) {
  model = std::make_shared<LinearModel>(indim, outdim);
}

} // namespace tests
} // namespace clf
