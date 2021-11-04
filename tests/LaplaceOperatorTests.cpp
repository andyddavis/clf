#include <gtest/gtest.h>

#include "clf/LaplaceOperator.hpp"
#include "clf/PolynomialBasis.hpp"
#include "clf/SinCosBasis.hpp"

namespace pt = boost::property_tree;
using namespace clf;

TEST(LaplaceOperatorTests, Evaluation) {
  // the in/output dimensions
  const std::size_t indim = 3, outdim = 2;

  // create the Laplace Operator
  auto laplace = std::make_shared<LaplaceOperator>(indim, outdim);

  // check in the input/output sizes
  EXPECT_EQ(laplace->inputDimension, indim);
  EXPECT_EQ(laplace->outputDimension, outdim);

  std::vector<std::shared_ptr<const BasisFunctions> > bases(outdim);

  pt::ptree polyBasisOptions;
  polyBasisOptions.put("InputDimension", indim);
  polyBasisOptions.put("Order", 2);
  bases[0] = PolynomialBasis::TotalOrderBasis(polyBasisOptions);

  pt::ptree trigBasisOptions;
  trigBasisOptions.put("InputDimension", indim);
  trigBasisOptions.put("Order", 2);
  bases[1] = SinCosBasis::TotalOrderBasis(trigBasisOptions);

  // pick a random point
  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  // the coefficients
  const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(bases[0]->NumBasisFunctions()+bases[1]->NumBasisFunctions());

  // try to evaluate the Laplace operator
  const Eigen::VectorXd modelEval = laplace->Operator(x, coefficients, bases);
  EXPECT_EQ(modelEval.size(), outdim);
  EXPECT_NEAR(modelEval(0), coefficients.head(bases[0]->NumBasisFunctions()).dot(bases[0]->EvaluateBasisFunctionDerivatives(x, 2).colwise().sum()), 1.0e-12);
  EXPECT_NEAR(modelEval(1), coefficients.tail(bases[1]->NumBasisFunctions()).dot(bases[1]->EvaluateBasisFunctionDerivatives(x, 2).colwise().sum()), 1.0e-12);
}
