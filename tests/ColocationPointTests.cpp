#include <gtest/gtest.h>

#include "clf/ColocationPoint.hpp"

namespace pt = boost::property_tree;
using namespace clf;

TEST(ColocationPointTests, Construction) {
  // input/output dimension
  const std::size_t indim = 3, outdim = 3;

  // create a model
  pt::ptree pt;
  pt.put("InputDimension", indim);
  pt.put("OutputDimension", outdim);
  auto model = std::make_shared<Model>(pt);

  const Eigen::VectorXd loc = Eigen::VectorXd::Random(indim);

  auto pnt = std::make_shared<ColocationPoint>(loc, model);
  EXPECT_NEAR((pnt->x-loc).norm(), 0.0, 1.0e-14);

  // the order of the total order polynomial basis
  const std::size_t order = 3;

  // create the support points
  pt.put("BasisFunctions", "Basis1, Basis2, Basis3");
  pt.put("Basis1.Type", "TotalOrderPolynomials");
  pt.put("Basis1.Order", order);
  pt.put("Basis2.Type", "TotalOrderPolynomials");
  pt.put("Basis2.Order", order);
  pt.put("Basis3.Type", "TotalOrderPolynomials");
  pt.put("Basis3.Order", order);
  auto supportPoint = SupportPoint::Construct(Eigen::VectorXd::Random(indim), pt);
  pnt->supportPoint = supportPoint;

  // evaluate the model with the random coefficeints
  Eigen::VectorXd x = Eigen::VectorXd::Random(indim);
  Eigen::VectorXd coeff = Eigen::VectorXd::Random(supportPoint->NumCoefficients());
  EXPECT_NEAR((pnt->Operator(x, coeff)-model->Operator(x, coeff, supportPoint->GetBasisFunctions())).norm(), 0.0, 1.0e-10);

  EXPECT_NEAR((pnt->OperatorJacobian(x, coeff)-model->OperatorJacobian(x, coeff, supportPoint->GetBasisFunctions())).norm(), 0.0, 1.0e-10);
}
