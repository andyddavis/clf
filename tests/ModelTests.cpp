#include <gtest/gtest.h>

#include "clf/LinearModel.hpp"
#include "clf/PolynomialBasis.hpp"
#include "clf/SinCosBasis.hpp"

#include "TestModels.hpp"

namespace pt = boost::property_tree;
using namespace clf;

TEST(ModelTests, RightHandSideEvaluationNotImplementedException) {
  // input/output dimension
  const std::size_t indim = 3, outdim = 5;

  // create a model
  pt::ptree pt;
  pt.put("InputDimension", indim);
  pt.put("OutputDimension", outdim);
  auto model = std::make_shared<LinearModel>(pt);
  EXPECT_TRUE(model->IsLinear());

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
  const std::size_t indim = 2, outdim = 2;

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
  auto model = std::make_shared<tests::TwoDimensionalAlgebraicModel>(pt);

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
  EXPECT_NEAR(modelEval(1), coefficients.tail(bases[1]->NumBasisFunctions()).dot(bases[1]->EvaluateBasisFunctions(x)) + std::pow(coefficients.tail(bases[1]->NumBasisFunctions()).dot(bases[1]->EvaluateBasisFunctions(x)), 2.0), 1.0e-12);

  // evaluate the identity operator jacobian by finite difference
  const Eigen::MatrixXd modelJacFD = model->OperatorJacobianByFD(x, coefficients, bases);
  EXPECT_EQ(modelJacFD.rows(), outdim);
  EXPECT_EQ(modelJacFD.cols(), coefficients.size());
  EXPECT_NEAR((modelJacFD.row(0).head(bases[0]->NumBasisFunctions())-bases[0]->EvaluateBasisFunctions(x).transpose()).norm(), 0.0, 1.0e-6);
  EXPECT_NEAR(modelJacFD.row(0).tail(bases[1]->NumBasisFunctions()).norm(), 0.0, 1.0e-12);

  EXPECT_NEAR((modelJacFD.row(1).tail(bases[1]->NumBasisFunctions())-bases[1]->EvaluateBasisFunctions(x).transpose()-(2.0*coefficients.tail(bases[1]->NumBasisFunctions()).dot(bases[1]->EvaluateBasisFunctions(x)))*bases[1]->EvaluateBasisFunctions(x).transpose()).norm(), 0.0, 1.0e-5);
  EXPECT_NEAR(modelJacFD.row(1).head(bases[0]->NumBasisFunctions()).norm(), 0.0, 1.0e-12);

  // evaluate the true jacobian using the default implementation
  const Eigen::MatrixXd modelJac = model->OperatorJacobian(x, coefficients, bases);
  EXPECT_EQ(modelJac.rows(), modelJacFD.rows());
  EXPECT_EQ(modelJac.cols(), modelJacFD.cols());
  EXPECT_NEAR((modelJac-modelJacFD).norm(), 0.0, 1.0e-5);

  // evaluate the true hessian using the default implementation
  const std::vector<Eigen::MatrixXd> modelHessFD = model->OperatorHessianByFD(x, coefficients, bases);
  const std::vector<Eigen::MatrixXd> modelHess = model->OperatorHessian(x, coefficients, bases);
  EXPECT_EQ(modelHessFD.size(), outdim);
  EXPECT_EQ(modelHess.size(), outdim);
  for( std::size_t i=0; i<outdim; ++i ) {
    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(coefficients.size(), coefficients.size());
    if( i==1 ) { 
      const Eigen::VectorXd phi = bases[1]->EvaluateBasisFunctions(x);
      expected.block(bases[0]->NumBasisFunctions(), bases[0]->NumBasisFunctions(), bases[1]->NumBasisFunctions(), bases[1]->NumBasisFunctions()) = 2.0*phi*phi.transpose();
    }

    EXPECT_EQ(modelHess[i].rows(), coefficients.size());
    EXPECT_EQ(modelHess[i].cols(), coefficients.size());
    EXPECT_NEAR((modelHess[i]-expected).norm(), 0.0, 1.0e-6);

    EXPECT_EQ(modelHessFD[i].rows(), coefficients.size());
    EXPECT_EQ(modelHessFD[i].cols(), coefficients.size());
    EXPECT_NEAR((modelHessFD[i]-expected).norm(), 0.0, 1.0e-6);
  }
}

TEST(ModelTests, RightHandSideEvaluationVectorValuedImplementation) {
  // input/output dimension
  const std::size_t indim = 3, outdim = 5;

  // create a model
  pt::ptree pt;
  pt.put("InputDimension", indim);
  pt.put("OutputDimension", outdim);
  auto model = std::make_shared<tests::ComponentWiseSquaredModel>(pt);
  EXPECT_FALSE(model->IsLinear());

  // check in the input/output sizes
  EXPECT_EQ(model->inputDimension, indim);
  EXPECT_EQ(model->outputDimension, outdim);

  // pick a random point
  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  // try to evaluate the right hand side
  const Eigen::VectorXd rhs = model->RightHandSide(x);
  EXPECT_EQ(rhs.size(), outdim);
  for( std::size_t i=0; i<outdim; ++i ) { EXPECT_NEAR(rhs(i), x.dot(x), 1.0e-12); }

  // evaluate the operator
  pt::ptree basisOptions;
  basisOptions.put("InputDimension", indim);
  std::vector<std::shared_ptr<const BasisFunctions> > bases(outdim, PolynomialBasis::TotalOrderBasis(basisOptions));
  const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(outdim*bases[0]->NumBasisFunctions());
  const Eigen::VectorXd identity = model->IdentityOperator(x, coefficients, bases);
  const Eigen::VectorXd u = model->Operator(x, coefficients, bases);
  EXPECT_NEAR((u-(identity.array()*identity.array()).matrix()).norm(), 0.0, 1.0e-10);

  // evaluate the jacobian of the operator
  const Eigen::MatrixXd jacFD = model->OperatorJacobianByFD(x, coefficients, bases);
  const Eigen::MatrixXd jac = model->OperatorJacobian(x, coefficients, bases);
  // the operator is not implemented so it should be exactly equal to the finite difference jacobian
  EXPECT_NEAR((jacFD-jac).norm(), 0.0, 1.0e-5);

  // evaluate the hessian of the operator
  const std::vector<Eigen::MatrixXd> trueHess = model->OperatorHessian(x, coefficients, bases);
  const std::vector<Eigen::MatrixXd> hessFD = model->OperatorHessianByFD(x, coefficients, bases);
  EXPECT_EQ(trueHess.size(), outdim);
  EXPECT_EQ(trueHess.size(), hessFD.size());
  for( std::size_t i=0; i<trueHess.size(); ++i ) {
    EXPECT_EQ(trueHess[i].rows(), coefficients.size());
    EXPECT_EQ(trueHess[i].cols(), coefficients.size());
    EXPECT_EQ(hessFD[i].rows(), coefficients.size());
    EXPECT_EQ(hessFD[i].cols(), coefficients.size());
    EXPECT_NEAR((trueHess[i]-hessFD[i]).norm(), 0.0, 1.0e-8);
  }
}

TEST(ModelTests, WrongNumberOfInputs) {
  // input/output dimension
  const std::size_t indim = 3, outdim = 5;

  // create a model
  pt::ptree pt;
  pt.put("InputDimension", indim);
  pt.put("OutputDimension", outdim);
  auto model = std::make_shared<tests::ComponentWiseSquaredModel>(pt);

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
  pt::ptree basisOptions;
  basisOptions.put("InputDimension", indim);
  std::vector<std::shared_ptr<const BasisFunctions> > bases(1, PolynomialBasis::TotalOrderBasis(basisOptions));
  const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(bases[0]->NumBasisFunctions());
  try {
    const Eigen::VectorXd output = model->Operator(x, coefficients, bases);
  } catch( exceptions::ModelHasWrongInputOutputDimensions const& exc ) {
    EXPECT_EQ(exc.type, exceptions::ModelHasWrongInputOutputDimensions::Type::INPUT);
    EXPECT_EQ(exc.func, exceptions::ModelHasWrongInputOutputDimensions::Function::OPERATOR);
    EXPECT_NE(exc.givendim, indim);
    EXPECT_EQ(exc.dim, indim);
  }

  // try to evaluate the Jacobian of the operator
  try {
    const Eigen::MatrixXd jac = model->OperatorJacobian(x, coefficients, bases);
  } catch( exceptions::ModelHasWrongInputOutputDimensions const& exc ) {
    EXPECT_EQ(exc.type, exceptions::ModelHasWrongInputOutputDimensions::Type::INPUT);
    EXPECT_EQ(exc.func, exceptions::ModelHasWrongInputOutputDimensions::Function::OPERATOR_JACOBIAN);
    EXPECT_NE(exc.givendim, indim);
    EXPECT_EQ(exc.dim, indim);
  }

  // try to evaluate the Hessian of the operator
  try {
    const std::vector<Eigen::MatrixXd> hess = model->OperatorHessian(x, coefficients, bases);
  } catch( exceptions::ModelHasWrongInputOutputDimensions const& exc ) {
    EXPECT_EQ(exc.type, exceptions::ModelHasWrongInputOutputDimensions::Type::INPUT);
    EXPECT_EQ(exc.func, exceptions::ModelHasWrongInputOutputDimensions::Function::OPERATOR_HESSIAN);
    EXPECT_NE(exc.givendim, indim);
    EXPECT_EQ(exc.dim, indim);
  }
}

TEST(ModelTests, WrongNumberOfOutputs) {
  // input/output dimension
  const std::size_t indim = 3, outdim = 5;

  // create a model
  pt::ptree pt;
  pt.put("InputDimension", indim);
  pt.put("OutputDimension", outdim);
  auto model = std::make_shared<tests::TestVectorValuedImplementationModelWrongOutputDim>(pt);

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
  pt::ptree basisOptions;
  basisOptions.put("InputDimension", indim);
  std::vector<std::shared_ptr<const BasisFunctions> > bases(outdim, PolynomialBasis::TotalOrderBasis(basisOptions));
  const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(outdim*bases[0]->NumBasisFunctions());
  try {
    const Eigen::VectorXd output = model->Operator(x, coefficients, bases);
  } catch( exceptions::ModelHasWrongInputOutputDimensions const& exc ) {
    EXPECT_EQ(exc.type, exceptions::ModelHasWrongInputOutputDimensions::Type::OUTPUT);
    EXPECT_EQ(exc.func, exceptions::ModelHasWrongInputOutputDimensions::Function::OPERATOR);
    EXPECT_NE(exc.givendim, outdim);
    EXPECT_EQ(exc.dim, outdim);
  }

  // try to evaluate the operator Jacobian
  try {
    const Eigen::MatrixXd output = model->OperatorJacobian(x, coefficients, bases);
  } catch( exceptions::ModelHasWrongInputOutputDimensions const& exc ) {
    EXPECT_EQ(exc.type, exceptions::ModelHasWrongInputOutputDimensions::Type::OUTPUT);
    EXPECT_EQ(exc.func, exceptions::ModelHasWrongInputOutputDimensions::Function::OPERATOR_JACOBIAN);
    EXPECT_NE(exc.givendim, outdim);
    EXPECT_EQ(exc.dim, outdim);
    EXPECT_NE(exc.givendimSecond, coefficients.size());
    EXPECT_EQ(exc.dimSecond, coefficients.size());
  }

  // try to evaluate the operator Hessian
  try {
    const std::vector<Eigen::MatrixXd> output = model->OperatorHessian(x, coefficients, bases);
  } catch( exceptions::ModelHasWrongInputOutputDimensions const& exc ) {
    EXPECT_EQ(exc.type, exceptions::ModelHasWrongInputOutputDimensions::Type::OUTPUT);
    EXPECT_EQ(exc.func, exceptions::ModelHasWrongInputOutputDimensions::Function::OPERATOR_HESSIAN_VECTOR);
    EXPECT_NE(exc.givendim, outdim);
    EXPECT_EQ(exc.dim, outdim);
  }
}

TEST(ModelTests, WrongNumberOfOutputsHessianCheck) {
  // input/output dimension
  const std::size_t indim = 3, outdim = 5;

  // create a model
  pt::ptree pt;
  pt.put("InputDimension", indim);
  pt.put("OutputDimension", outdim);
  auto model = std::make_shared<tests::TestHessianImplementationModelWrongOutputDim>(pt);

  // check in the input/output sizes
  EXPECT_EQ(model->inputDimension, indim);
  EXPECT_EQ(model->outputDimension, outdim);

  // pick a random point
  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  // try to evaluate the operator Hessian
  pt::ptree basisOptions;
  basisOptions.put("InputDimension", indim);
  std::vector<std::shared_ptr<const BasisFunctions> > bases(outdim, PolynomialBasis::TotalOrderBasis(basisOptions));
  const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(outdim*bases[0]->NumBasisFunctions());
  try {
    const std::vector<Eigen::MatrixXd> output = model->OperatorHessian(x, coefficients, bases);
  } catch( exceptions::ModelHasWrongInputOutputDimensions const& exc ) {
    EXPECT_EQ(exc.type, exceptions::ModelHasWrongInputOutputDimensions::Type::OUTPUT);
    EXPECT_EQ(exc.func, exceptions::ModelHasWrongInputOutputDimensions::Function::OPERATOR_HESSIAN_MATRIX);
    EXPECT_NE(exc.givendim, coefficients.size());
    EXPECT_EQ(exc.dim, coefficients.size());
    EXPECT_NE(exc.givendimSecond, coefficients.size());
    EXPECT_EQ(exc.dimSecond, coefficients.size());
  }
}

TEST(ModelTests, FunctionDerivativeWrongInputDimension) {
  // input/output dimension
  const std::size_t indim = 3, outdim = 5;

  // create a model
  pt::ptree pt;
  pt.put("InputDimension", indim);
  pt.put("OutputDimension", outdim);
  auto model = std::make_shared<tests::TestDifferentialOperatorImplementationModel>(pt);

  // check in the input/output sizes
  EXPECT_EQ(model->inputDimension, indim);
  EXPECT_EQ(model->outputDimension, outdim);

  // pick a random point
  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim+1);

  // evaluate the function derivatives
  pt::ptree basisOptions;
  basisOptions.put("InputDimension", indim);
  std::vector<std::shared_ptr<const BasisFunctions> > bases(outdim, PolynomialBasis::TotalOrderBasis(basisOptions));
  const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(outdim*bases[0]->NumBasisFunctions());

  // try to evaluate the right hand side
  try {
    const double du0dx0 = model->FunctionDerivative(x, coefficients, bases, 1, 0, 0);
  } catch( exceptions::ModelHasWrongInputOutputDimensions const& exc ) {
    EXPECT_EQ(exc.type, exceptions::ModelHasWrongInputOutputDimensions::Type::INPUT);
    EXPECT_EQ(exc.func, exceptions::ModelHasWrongInputOutputDimensions::Function::FUNCTION_DERIVATIVE);
    EXPECT_EQ(exc.givendim, indim+1);
    EXPECT_EQ(exc.dim, indim);
  }
}

TEST(ModelTests, FunctionDerivativeEvaluation) {
  // input/output dimension
  const std::size_t indim = 3, outdim = 5;

  // create a model
  pt::ptree pt;
  pt.put("InputDimension", indim);
  pt.put("OutputDimension", outdim);
  auto model = std::make_shared<tests::TestDifferentialOperatorImplementationModel>(pt);

  // check in the input/output sizes
  EXPECT_EQ(model->inputDimension, indim);
  EXPECT_EQ(model->outputDimension, outdim);

  // pick a random point
  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  // evaluate the function derivatives
  pt::ptree basisOptions;
  basisOptions.put("InputDimension", indim);
  std::vector<std::shared_ptr<const BasisFunctions> > bases(outdim, PolynomialBasis::TotalOrderBasis(basisOptions));
  const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(outdim*bases[0]->NumBasisFunctions());

  // compute the derivative
  const std::size_t k = 1; // the order of the derivative
  std::size_t coeffInd = 0;
  for( std::size_t j=0; j<outdim; ++j ) {
    for( std::size_t i=0; i<indim; ++i ) {
      const double dujdxi = model->FunctionDerivative(x, coefficients, bases, j, i, k);

      const double expected = coefficients.segment(coeffInd, bases[j]->NumBasisFunctions()).dot(bases[j]->EvaluateBasisFunctionDerivatives(x, i, k));

      EXPECT_NEAR(dujdxi, expected, 1.0e-10);
    }
    coeffInd += bases[j]->NumBasisFunctions();
  }
}

TEST(ModelTests, DifferentialOperator) {
  // input/output dimension
  const std::size_t indim = 3, outdim = 5;

  // create a model
  pt::ptree pt;
  pt.put("InputDimension", indim);
  pt.put("OutputDimension", outdim);
  auto model = std::make_shared<tests::TestDifferentialOperatorImplementationModel>(pt);

  // check in the input/output sizes
  EXPECT_EQ(model->inputDimension, indim);
  EXPECT_EQ(model->outputDimension, outdim);

  // pick a random point
  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  // evaluate the operator
  pt::ptree basisOptions;
  basisOptions.put("InputDimension", indim);
  std::vector<std::shared_ptr<const BasisFunctions> > bases(outdim, PolynomialBasis::TotalOrderBasis(basisOptions));
  const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(outdim*bases[0]->NumBasisFunctions());
  const Eigen::VectorXd identity = model->IdentityOperator(x, coefficients, bases);
  const Eigen::VectorXd Lu = model->Operator(x, coefficients, bases);
  Eigen::VectorXd expected = Eigen::VectorXd::Zero(outdim);
  for( std::size_t out=0; out<outdim; ++out ) {
    for( std::size_t in=0; in<indim; ++in ) { expected(out) += identity(out)*model->FunctionDerivative(x, coefficients, bases, out, in, 1); }
  }
  EXPECT_NEAR((Lu-expected).norm(), 0.0, 1.0e-10);

  // evaluate the jacobian of the operator
  const Eigen::MatrixXd jacFD = model->OperatorJacobianByFD(x, coefficients, bases);
  const Eigen::MatrixXd jac = model->OperatorJacobian(x, coefficients, bases);
  EXPECT_NEAR((jacFD-jac).norm(), 0.0, 1.0e-5);

  // evaluate the hessian of the operator
  const std::vector<Eigen::MatrixXd> hess = model->OperatorHessian(x, coefficients, bases);
  const std::vector<Eigen::MatrixXd> hessFD = model->OperatorHessianByFD(x, coefficients, bases);
  EXPECT_EQ(hess.size(), outdim);
  EXPECT_EQ(hess.size(), hessFD.size());
  for( std::size_t i=0; i<hess.size(); ++i ) {
    EXPECT_EQ(hess[i].rows(), coefficients.size());
    EXPECT_EQ(hess[i].cols(), coefficients.size());
    EXPECT_EQ(hessFD[i].rows(), coefficients.size());
    EXPECT_EQ(hessFD[i].cols(), coefficients.size());
    EXPECT_NEAR((hess[i]-hessFD[i]).norm(), 0.0, 1.0e-8);
  }
}
