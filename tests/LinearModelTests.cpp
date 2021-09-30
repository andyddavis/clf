#include <gtest/gtest.h>

#include "clf/LinearModel.hpp"
#include "clf/PolynomialBasis.hpp"

namespace pt = boost::property_tree;
using namespace clf;

TEST(LinearModelTests, Identity) {
  const std::size_t indim = 3;
  const std::size_t outdim = 2;

  pt::ptree options;
  options.put("InputDimension", indim);
  options.put("OutputDimension", outdim);
  auto model = std::make_shared<LinearModel>(options);

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
