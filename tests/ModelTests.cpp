#include <gtest/gtest.h>

#include "clf/Model.hpp"
#include "clf/PolynomialBasis.hpp"
#include "clf/SinCosBasis.hpp"

namespace pt = boost::property_tree;
using namespace clf;

TEST(ModelTests, RightHandSideEvaluationNotImplementedException) {
  // input/output dimension
  const std::size_t indim = 3, outdim = 5;

  // create a model
  pt::ptree pt;
  pt.put("InputDimension", indim);
  pt.put("OutputDimension", outdim);
  auto model = std::make_shared<Model>(pt);

  // check in the input/output sizes
  EXPECT_EQ(model->inputDimension, indim);
  EXPECT_EQ(model->outputDimension, outdim);

  // pick a random point
  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  // try to evaluate the right hand side
  try {
    const Eigen::VectorXd rhs = model->RightHandSide(x);
  } catch( exceptions::ModelHasNotImplemented const& exc ) {
    EXPECT_EQ(exc.type, exceptions::ModelHasNotImplemented::Type::BOTH);
    EXPECT_EQ(exc.func, exceptions::ModelHasNotImplemented::Function::RHS);
  }
}

TEST(ModelTests, OperatorEvaluationDefaultImplementation) {
  // input/output dimension
  const std::size_t indim = 3, outdim = 2;

  std::vector<std::shared_ptr<const BasisFunctions> > bases(outdim);

  pt::ptree polyBasisOptions;
  polyBasisOptions.put("InputDimension", indim);
  polyBasisOptions.put("Order", 2);
  bases[0] = PolynomialBasis::TotalOrderBasis(polyBasisOptions);

  pt::ptree trigBasisOptions;
  trigBasisOptions.put("InputDimension", indim);
  trigBasisOptions.put("Order", 2);
  bases[1] = SinCosBasis::TotalOrderBasis(trigBasisOptions);

  // create a model
  pt::ptree pt;
  pt.put("InputDimension", indim);
  pt.put("OutputDimension", outdim);
  auto model = std::make_shared<Model>(pt);

  // check in the input/output sizes
  EXPECT_EQ(model->inputDimension, indim);
  EXPECT_EQ(model->outputDimension, outdim);

  // pick a random point
  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  // the coefficients
  const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(bases[0]->NumBasisFunctions()+bases[1]->NumBasisFunctions());

  // try to evaluate the operator---the default is the identity operator
  const Eigen::VectorXd modelEval = model->Operator(x, coefficients, bases);
  EXPECT_EQ(modelEval.size(), outdim);
  EXPECT_NEAR(modelEval(0), coefficients.head(bases[0]->NumBasisFunctions()).dot(bases[0]->EvaluateBasisFunctions(x)), 1.0e-12);
  EXPECT_NEAR(modelEval(1), coefficients.tail(bases[1]->NumBasisFunctions()).dot(bases[1]->EvaluateBasisFunctions(x)), 1.0e-12);
}

class TestVectorValuedImplementationModel : public Model {
public:

  inline TestVectorValuedImplementationModel(pt::ptree const& pt) : Model(pt) {}

  virtual ~TestVectorValuedImplementationModel() = default;
protected:

  /**
  @param[in] x The point \f$x \in \Omega \f$
  \return The evaluation of \f$f(x)\f$
  */
  inline virtual Eigen::VectorXd RightHandSideVectorImpl(Eigen::VectorXd const& x) const override { return Eigen::VectorXd::Constant(outputDimension, x.prod()); }

  /**
  */
  inline virtual Eigen::VectorXd OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    return Eigen::VectorXd();
  }

private:
};

TEST(ModelTests, RightHandSideEvaluationVectorValuedImplementation) {
  // input/output dimension
  const std::size_t indim = 3, outdim = 5;

  // create a model
  pt::ptree pt;
  pt.put("InputDimension", indim);
  pt.put("OutputDimension", outdim);
  auto model = std::make_shared<TestVectorValuedImplementationModel>(pt);

  // check in the input/output sizes
  EXPECT_EQ(model->inputDimension, indim);
  EXPECT_EQ(model->outputDimension, outdim);

  // pick a random point
  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  // try to evaluate the right hand side
  const Eigen::VectorXd rhs = model->RightHandSide(x);
  EXPECT_EQ(rhs.size(), outdim);
  for( std::size_t i=0; i<outdim; ++i ) { EXPECT_NEAR(rhs(i), x.prod(), 1.0e-12); }
}

TEST(ModelTests, RightHandSideEvaluationWrongNumberOfInputs) {
  // input/output dimension
  const std::size_t indim = 3, outdim = 5;

  // create a model
  pt::ptree pt;
  pt.put("InputDimension", indim);
  pt.put("OutputDimension", outdim);
  auto model = std::make_shared<TestVectorValuedImplementationModel>(pt);

  // check in the input/output sizes
  EXPECT_EQ(model->inputDimension, indim);
  EXPECT_EQ(model->outputDimension, outdim);

  // pick a random point
  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim+1);

  // try to evaluate the right hand side
  try {
    const Eigen::VectorXd rhs = model->RightHandSide(x);
  } catch( exceptions::ModelHasWrongInputOutputDimensions const& exc ) {
    EXPECT_EQ(exc.type, exceptions::ModelHasWrongInputOutputDimensions::Type::INPUT);
    EXPECT_EQ(exc.func, exceptions::ModelHasWrongInputOutputDimensions::Function::RHS);
    EXPECT_NE(exc.givendim, indim);
    EXPECT_EQ(exc.dim, indim);
  }

  // try to evaluate the operator
  try {
    pt::ptree basisOptions;
    basisOptions.put("InputDimension", indim);
    std::vector<std::shared_ptr<const BasisFunctions> > bases(1, PolynomialBasis::TotalOrderBasis(basisOptions));
    const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(bases[0]->NumBasisFunctions());
    const Eigen::VectorXd output = model->Operator(x, coefficients, bases);
  } catch( exceptions::ModelHasWrongInputOutputDimensions const& exc ) {
    EXPECT_EQ(exc.type, exceptions::ModelHasWrongInputOutputDimensions::Type::INPUT);
    EXPECT_EQ(exc.func, exceptions::ModelHasWrongInputOutputDimensions::Function::OPERATOR);
    EXPECT_NE(exc.givendim, indim);
    EXPECT_EQ(exc.dim, indim);
  }
}

class TestVectorValuedImplementationModelWrongOutputDim : public Model {
public:

  inline TestVectorValuedImplementationModelWrongOutputDim(pt::ptree const& pt) : Model(pt) {}

  virtual ~TestVectorValuedImplementationModelWrongOutputDim() = default;
protected:

  /**
  @param[in] x The point \f$x \in \Omega \f$
  \return The evaluation of \f$f(x)\f$
  */
  inline virtual Eigen::VectorXd RightHandSideVectorImpl(Eigen::VectorXd const& x) const override { return Eigen::VectorXd::Constant(outputDimension+1, x.prod()); }

  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is devided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  inline virtual Eigen::VectorXd OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override { return Eigen::VectorXd::Constant(outputDimension+1, x.prod()); }

private:
};

TEST(ModelTests, RightHandSideEvaluationWrongNumberOfOutputs) {
  // input/output dimension
  const std::size_t indim = 3, outdim = 5;

  // create a model
  pt::ptree pt;
  pt.put("InputDimension", indim);
  pt.put("OutputDimension", outdim);
  auto model = std::make_shared<TestVectorValuedImplementationModelWrongOutputDim>(pt);

  // check in the input/output sizes
  EXPECT_EQ(model->inputDimension, indim);
  EXPECT_EQ(model->outputDimension, outdim);

  // pick a random point
  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  // try to evaluate the right hand side
  try {
    const Eigen::VectorXd rhs = model->RightHandSide(x);
  } catch( exceptions::ModelHasWrongInputOutputDimensions const& exc ) {
    EXPECT_EQ(exc.type, exceptions::ModelHasWrongInputOutputDimensions::Type::OUTPUT);
    EXPECT_EQ(exc.func, exceptions::ModelHasWrongInputOutputDimensions::Function::RHS);
    EXPECT_NE(exc.givendim, outdim);
    EXPECT_EQ(exc.dim, outdim);
  }

  // try to evaluate the operator
  try {
    pt::ptree basisOptions;
    basisOptions.put("InputDimension", indim);
    std::vector<std::shared_ptr<const BasisFunctions> > bases(outdim, PolynomialBasis::TotalOrderBasis(basisOptions));
    const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(outdim*bases[0]->NumBasisFunctions());
    const Eigen::VectorXd output = model->Operator(x, coefficients, bases);
  } catch( exceptions::ModelHasWrongInputOutputDimensions const& exc ) {
    EXPECT_EQ(exc.type, exceptions::ModelHasWrongInputOutputDimensions::Type::OUTPUT);
    EXPECT_EQ(exc.func, exceptions::ModelHasWrongInputOutputDimensions::Function::OPERATOR);
    EXPECT_NE(exc.givendim, outdim);
    EXPECT_EQ(exc.dim, outdim);
  }
}

class TestComponentWiseImplementationModel : public Model {
public:

  inline TestComponentWiseImplementationModel(pt::ptree const& pt) : Model(pt) {}

  virtual ~TestComponentWiseImplementationModel() = default;
protected:

  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] outind Return this component of the evaluation of \f$f\f$
  \return The component of \f$f(x)\f$ corresponding to <tt>outind</tt>
  */
  inline virtual double RightHandSideComponentImpl(Eigen::VectorXd const& x, std::size_t const outind) const override { return x.prod(); }

private:
};

TEST(ModelTests, RightHandSideEvaluationComponentWiseImplementation) {
  // input/output dimension
  const std::size_t indim = 3, outdim = 5;

  // create a model
  pt::ptree pt;
  pt.put("InputDimension", indim);
  pt.put("OutputDimension", outdim);
  auto model = std::make_shared<TestComponentWiseImplementationModel>(pt);

  // check in the input/output sizes
  EXPECT_EQ(model->inputDimension, indim);
  EXPECT_EQ(model->outputDimension, outdim);

  // pick a random point
  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  // try to evaluate the right hand side
  const Eigen::VectorXd rhs = model->RightHandSide(x);
  EXPECT_EQ(rhs.size(), outdim);
  for( std::size_t i=0; i<outdim; ++i ) { EXPECT_NEAR(rhs(i), x.prod(), 1.0e-12); }
}
