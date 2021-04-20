#include <gtest/gtest.h>

#include <cstdlib>

#include "clf/SinCosBasis.hpp"

namespace pt = boost::property_tree;
using namespace muq::Utilities;
using namespace clf;

TEST(SinCosBasisTests, Evaluation) {
  // the input dimension
  const std::size_t dim = 5;

  // the multi-index set
  auto multis = std::make_shared<MultiIndexSet>(dim);

  { // add constant multi-indices
    const Eigen::RowVectorXi ind = Eigen::RowVectorXi::Zero(dim);
    multis->AddActive(std::make_shared<MultiIndex>(ind));
  }

  // add sin and cos basis function
  for( std::size_t i=0; i<dim; ++i ) {
    Eigen::RowVectorXi ind = Eigen::RowVectorXi::Zero(dim);
    ind(i) = 1.0;
    multis->AddActive(std::make_shared<MultiIndex>(ind));

    ind(i) = 2.0;
    multis->AddActive(std::make_shared<MultiIndex>(ind));
  }

  pt::ptree pt;
  pt.put("Type", "SinCosBasis");

  auto basis = BasisFunctions::Construct(multis, pt);
  EXPECT_TRUE(basis);
  auto sincosBasis = std::dynamic_pointer_cast<SinCosBasis>(basis);
  EXPECT_TRUE(sincosBasis);

  EXPECT_EQ(basis->NumBasisFunctions(), 1+2*dim);

  // choose a random point
  const Eigen::VectorXd x = Eigen::VectorXd::Random(dim);

  // evaluate the constant basis function
  EXPECT_DOUBLE_EQ(basis->EvaluateBasisFunction(x, 0), 1.0);
  for( std::size_t d=0; d<dim; ++d ) { EXPECT_DOUBLE_EQ(basis->EvaluateBasisFunctionDerivative(x, 0, d, std::rand()), 0.0); }

  // evaluate the sin/cos basis function
  const Eigen::VectorXd phi = basis->EvaluateBasisFunctions(x);
  const Eigen::MatrixXd dphidxk = basis->EvaluateBasisFunctionDerivatives(x, 10);
  EXPECT_EQ(phi.size(), 2*dim+1);
  EXPECT_EQ(dphidxk.rows(), x.size());
  EXPECT_EQ(dphidxk.cols(), 2*dim+1);
  EXPECT_NEAR(dphidxk.col(0).norm(), 0.0, 1.0e-12);
  for( std::size_t i=0; i<dim; ++i ) {
    EXPECT_DOUBLE_EQ(phi(2*(i+1)-1), std::sin(M_PI*x(i)));
    EXPECT_DOUBLE_EQ(phi(2*(i+1)), std::cos(M_PI*x(i)));

    EXPECT_DOUBLE_EQ(basis->EvaluateBasisFunction(x, 2*(i+1)-1), std::sin(M_PI*x(i)));
    EXPECT_DOUBLE_EQ(basis->EvaluateBasisFunction(x, 2*(i+1)), std::cos(M_PI*x(i)));

    EXPECT_DOUBLE_EQ(basis->EvaluateBasisFunctionDerivative(x, 2*(i+1)-1, i, 0), std::sin(M_PI*x(i)));
    EXPECT_DOUBLE_EQ(basis->EvaluateBasisFunctionDerivative(x, 2*(i+1)-1, i, 1), M_PI*std::cos(M_PI*x(i)));
    EXPECT_DOUBLE_EQ(basis->EvaluateBasisFunctionDerivative(x, 2*(i+1)-1, i, 2), -M_PI*M_PI*std::sin(M_PI*x(i)));
    EXPECT_DOUBLE_EQ(basis->EvaluateBasisFunctionDerivative(x, 2*(i+1)-1, i, 3), -M_PI*M_PI*M_PI*std::cos(M_PI*x(i)));
    EXPECT_DOUBLE_EQ(basis->EvaluateBasisFunctionDerivative(x, 2*(i+1)-1, i, 4), M_PI*M_PI*M_PI*M_PI*std::sin(M_PI*x(i)));

    EXPECT_DOUBLE_EQ(basis->EvaluateBasisFunctionDerivative(x, 2*(i+1), i, 0), std::cos(M_PI*x(i)));
    EXPECT_DOUBLE_EQ(basis->EvaluateBasisFunctionDerivative(x, 2*(i+1), i, 1), -M_PI*std::sin(M_PI*x(i)));
    EXPECT_DOUBLE_EQ(basis->EvaluateBasisFunctionDerivative(x, 2*(i+1), i, 2), -M_PI*M_PI*std::cos(M_PI*x(i)));
    EXPECT_DOUBLE_EQ(basis->EvaluateBasisFunctionDerivative(x, 2*(i+1), i, 3), M_PI*M_PI*M_PI*std::sin(M_PI*x(i)));
    EXPECT_DOUBLE_EQ(basis->EvaluateBasisFunctionDerivative(x, 2*(i+1), i, 4), M_PI*M_PI*M_PI*M_PI*std::cos(M_PI*x(i)));
  }
}

TEST(TotalOrderSineCosineBasisTests, Construction) {
  const std::size_t dim = 1, order = 5;

  pt::ptree pt;
  pt.put("InputDimension", dim);
  pt.put("Order", order);

  auto basis = SinCosBasis::TotalOrderBasis(pt);
  EXPECT_TRUE(basis);

  EXPECT_EQ(basis->NumBasisFunctions(), 1+2*order);
}
