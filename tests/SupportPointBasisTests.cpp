#include <gtest/gtest.h>

#include "clf/SupportPointBasis.hpp"
#include "clf/SupportPoint.hpp"
#include "clf/LinearModel.hpp"

namespace pt = boost::property_tree;
using namespace clf;

TEST(SupportPointBasisTests, Construct) {
  // the input and output dimensions
  const std::size_t indim = 4, outdim = 1;

  pt::ptree modelOptions;
  modelOptions.put("InputDimension", indim);
  modelOptions.put("OutputDimension", outdim);
  auto model = std::make_shared<LinearModel>(modelOptions);

  // choose a random location
  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  // create the support point
  pt::ptree suppPointOptions;
  suppPointOptions.put("BasisFunctions", "Basis");
  suppPointOptions.put("Basis.Type", "TotalOrderPolynomials");
  suppPointOptions.put("Basis.Radius", 0.5);
  auto point = SupportPoint::Construct(x, model, suppPointOptions);

  // the basis is a support point basis
  auto basis = std::dynamic_pointer_cast<const SupportPointBasis>(point->GetBasisFunctions() [0]);
  EXPECT_TRUE(basis);
  EXPECT_DOUBLE_EQ(basis->Radius(), 0.5);
  const double newdelta = 0.75;
  basis->SetRadius(newdelta);
  EXPECT_DOUBLE_EQ(basis->Radius(), newdelta);

  // chose a nearby point and compute the local coordinate
  const Eigen::VectorXd y = point->x + 0.1*newdelta*Eigen::VectorXd::Random(indim);
  const Eigen::VectorXd yhat = basis->LocalCoordinate(y);
  EXPECT_NEAR(((y-point->x)/newdelta - yhat).norm(), 0.0, 1.0e-10);
  EXPECT_NEAR((basis->GlobalCoordinate(yhat) - y).norm(), 0.0, 1.0e-10);

  // make sure the implementation is correct
  const Eigen::VectorXd phihat = basis->basis->EvaluateBasisFunctions(yhat);
  const Eigen::VectorXd phi = basis->EvaluateBasisFunctions(y);
  EXPECT_NEAR((phihat-phi).norm(), 0.0, 1.0e-12);
  for( std::size_t i=0; i<10; ++i ) {
    const Eigen::MatrixXd phihatDeriv = basis->basis->EvaluateBasisFunctionDerivatives(yhat, i)/std::pow(newdelta, (double)i);
    const Eigen::MatrixXd phiDeriv = basis->EvaluateBasisFunctionDerivatives(y, i);
    EXPECT_NEAR((phihatDeriv-phiDeriv).norm(), 0.0, 1.0e-12);
  }
}
