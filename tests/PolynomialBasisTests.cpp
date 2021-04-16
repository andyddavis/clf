#include <gtest/gtest.h>

#include "clf/PolynomialBasis.hpp"

namespace pt = boost::property_tree;
using namespace muq::Utilities;
using namespace muq::Approximation;
using namespace clf;

class PolynomialBasisTests : public::testing::Test {
protected:
  /// Set up information to test the support point
  virtual void SetUp() override {
    { // add constant multi-indices
      const Eigen::RowVectorXi ind = Eigen::RowVectorXi::Zero(dim);
      multis->AddActive(std::make_shared<MultiIndex>(ind));
    }

    // add linear and quadratic basis functions
    for( std::size_t i=0; i<dim; ++i ) {
      Eigen::RowVectorXi ind = Eigen::RowVectorXi::Zero(dim);
      ind(i) = 1.0;
      multis->AddActive(std::make_shared<MultiIndex>(ind));

      ind(i) = 2.0;
      multis->AddActive(std::make_shared<MultiIndex>(ind));
    }
  }

  /// Set up information to test the support point
  virtual void TearDown() override {
    pt::ptree pt;
    pt.put("Type", "PolynomialBasis");
    pt.put("ScalarBasis", basisName);

    auto basis = BasisFunctions::Construct(multis, pt);
    EXPECT_TRUE(basis);
    auto polyBasis = std::dynamic_pointer_cast<PolynomialBasis>(basis);
    EXPECT_TRUE(polyBasis);

    EXPECT_EQ(basis->NumBasisFunctions(), 1+2*dim);

    // choose a random point
    const Eigen::VectorXd x = Eigen::VectorXd::Random(dim);

    // evaluate the constant basis function
    EXPECT_DOUBLE_EQ(basis->EvaluateBasisFunction(0, x), 1.0);

    auto poly = IndexedScalarBasis::Construct(basisName);

    // evaluate the polynomial basis functions
    const Eigen::VectorXd phi = basis->EvaluateBasisFunctions(x);
    for( std::size_t i=0; i<dim; ++i ) {
      EXPECT_DOUBLE_EQ(phi(2*(i+1)-1), poly->BasisEvaluate(1, x(i)));
      EXPECT_DOUBLE_EQ(phi(2*(i+1)), poly->BasisEvaluate(2, x(i)));

      EXPECT_DOUBLE_EQ(basis->EvaluateBasisFunction(2*(i+1)-1, x), poly->BasisEvaluate(1, x(i)));
      EXPECT_DOUBLE_EQ(basis->EvaluateBasisFunction(2*(i+1), x), poly->BasisEvaluate(2, x(i)));
    }
  }

  // the input dimension
  const std::size_t dim = 5;

  // the multi-index set
  std::shared_ptr<MultiIndexSet> multis = std::make_shared<MultiIndexSet>(dim);

  std::string basisName;
};

TEST_F(PolynomialBasisTests, LegendreEvaluation) {
  basisName = "Legendre";
}

TEST_F(PolynomialBasisTests, JacobiEvaluation) {
  basisName = "Jacobi";
}

TEST_F(PolynomialBasisTests, LaguerreEvaluation) {
  basisName = "Laguerre";
}

TEST_F(PolynomialBasisTests, ProbabilistHermiteEvaluation) {
  basisName = "ProbabilistHermite";
}

TEST_F(PolynomialBasisTests, PhysicistHermiteEvaluation) {
  basisName = "PhysicistHermite";
}

TEST_F(PolynomialBasisTests, MonomialEvaluation) {
  basisName = "Monomial";
}

TEST(TotalOrderPolynomialBasisTests, Construction) {
  const std::size_t dim = 1, order = 5;

  pt::ptree pt;
  pt.put("InputDimension", dim);
  pt.put("Order", order);

  auto basis = PolynomialBasis::TotalOrderBasis(pt);
  EXPECT_TRUE(basis);

  EXPECT_EQ(basis->NumBasisFunctions(), 1+order);
}
