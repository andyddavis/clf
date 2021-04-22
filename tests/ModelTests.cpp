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

  // evaluate the identity operator jacobian by finite difference
  const Eigen::MatrixXd modelJacFD = model->OperatorJacobianByFD(x, coefficients, bases);
  EXPECT_EQ(modelJacFD.rows(), outdim);
  EXPECT_EQ(modelJacFD.cols(), coefficients.size());
  EXPECT_NEAR((modelJacFD.row(0).head(bases[0]->NumBasisFunctions())-bases[0]->EvaluateBasisFunctions(x).transpose()).norm(), 0.0, 1.0e-6);
  EXPECT_NEAR((modelJacFD.row(0).tail(bases[1]->NumBasisFunctions())-Eigen::RowVectorXd::Zero(bases[1]->NumBasisFunctions())).norm(), 0.0, 1.0e-12);
  EXPECT_NEAR((modelJacFD.row(1).tail(bases[1]->NumBasisFunctions())-bases[1]->EvaluateBasisFunctions(x).transpose()).norm(), 0.0, 1.0e-6);
  EXPECT_NEAR((modelJacFD.row(1).head(bases[0]->NumBasisFunctions())-Eigen::RowVectorXd::Zero(bases[0]->NumBasisFunctions())).norm(), 0.0, 1.0e-12);

  // evaluate the true jacobian using the default implementation
  const Eigen::MatrixXd modelJac = model->OperatorJacobian(x, coefficients, bases);
  EXPECT_EQ(modelJac.rows(), outdim);
  EXPECT_EQ(modelJac.cols(), coefficients.size());
  EXPECT_NEAR((modelJac.row(0).head(bases[0]->NumBasisFunctions())-bases[0]->EvaluateBasisFunctions(x).transpose()).norm(), 0.0, 1.0e-12);
  EXPECT_NEAR((modelJac.row(0).tail(bases[1]->NumBasisFunctions())-Eigen::RowVectorXd::Zero(bases[1]->NumBasisFunctions())).norm(), 0.0, 1.0e-12);
  EXPECT_NEAR((modelJac.row(1).tail(bases[1]->NumBasisFunctions())-bases[1]->EvaluateBasisFunctions(x).transpose()).norm(), 0.0, 1.0e-12);
  EXPECT_NEAR((modelJac.row(1).head(bases[0]->NumBasisFunctions())-Eigen::RowVectorXd::Zero(bases[0]->NumBasisFunctions())).norm(), 0.0, 1.0e-12);

  // evaluate the true hessian using the default implementation
  const std::vector<Eigen::MatrixXd> modelHessFD = model->OperatorHessianByFD(x, coefficients, bases);
  const std::vector<Eigen::MatrixXd> modelHess = model->OperatorHessian(x, coefficients, bases);
  EXPECT_EQ(modelHessFD.size(), coefficients.size());
  EXPECT_EQ(modelHess.size(), coefficients.size());
  for( std::size_t i=0; i<coefficients.size(); ++i ) {
    EXPECT_EQ(modelHess[i].rows(), outdim);
    EXPECT_EQ(modelHess[i].cols(), coefficients.size());
    EXPECT_DOUBLE_EQ(modelHess[i].norm(), 0.0);

    EXPECT_EQ(modelHessFD[i].rows(), outdim);
    EXPECT_EQ(modelHessFD[i].cols(), coefficients.size());
    EXPECT_DOUBLE_EQ(modelHessFD[i].norm(), 0.0);
  }
}

class TestVectorValuedImplementationModel : public Model {
public:

  inline TestVectorValuedImplementationModel(pt::ptree const& pt) : Model(pt) {}

  virtual ~TestVectorValuedImplementationModel() = default;

  Eigen::MatrixXd TrueJacobian(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
    Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(outputDimension, coefficients.size());

    std::size_t ind = 0;
    for( std::size_t i=0; i<outputDimension; ++i ) {
      std::size_t basisSize = bases[i]->NumBasisFunctions();
      const Eigen::VectorXd phi = bases[i]->EvaluateBasisFunctions(x);
      assert(phi.size()==basisSize);
      jac.row(i).segment(ind, basisSize) = phi.dot(coefficients.segment(ind, basisSize))*phi;
      ind += basisSize;
    }

    return 2.0*jac;
  }

protected:

  /**
  @param[in] x The point \f$x \in \Omega \f$
  \return The evaluation of \f$f(x)\f$
  */
  inline virtual Eigen::VectorXd RightHandSideVectorImpl(Eigen::VectorXd const& x) const override { return Eigen::VectorXd::Constant(outputDimension, x.prod()); }

  /// The operator \f$\mathcal{L}(u) = u^2\f$
  /**
  */
  inline virtual Eigen::VectorXd OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    assert(bases.size()==outputDimension);
    Eigen::VectorXd output = IdentityOperator(x, coefficients, bases);
    output = output.array()*output.array();

    return output;
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

  // evaluate the operator
  pt::ptree basisOptions;
  basisOptions.put("InputDimension", indim);
  std::vector<std::shared_ptr<const BasisFunctions> > bases(outdim, PolynomialBasis::TotalOrderBasis(basisOptions));
  const Eigen::VectorXd coefficients = Eigen::VectorXd::Random(outdim*bases[0]->NumBasisFunctions());
  const Eigen::VectorXd identity = model->IdentityOperator(x, coefficients, bases);
  const Eigen::VectorXd u = model->Operator(x, coefficients, bases);
  EXPECT_NEAR((u-(identity.array()*identity.array()).matrix()).norm(), 0.0, 1.0e-10);

  // evaluate the jacobian of the operator
  const Eigen::MatrixXd trueJac = model->TrueJacobian(x, coefficients, bases);
  const Eigen::MatrixXd jacFD = model->OperatorJacobianByFD(x, coefficients, bases);
  const Eigen::MatrixXd jac = model->OperatorJacobian(x, coefficients, bases);
  // the operator is not implemented so it should be exactly equal to the finite difference jacobian
  EXPECT_NEAR((jac-trueJac).norm(), 0.0, 1.0e-5);
  EXPECT_NEAR((jacFD-jac).norm(), 0.0, 1.0e-10);
}

TEST(ModelTests, WrongNumberOfInputs) {
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

  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is devided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  inline virtual Eigen::MatrixXd OperatorJacobianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override { return Eigen::MatrixXd(outputDimension+1, coefficients.size()+1); }

  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is devided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  inline virtual std::vector<Eigen::MatrixXd> OperatorHessianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override { return std::vector<Eigen::MatrixXd>(coefficients.size()+1); }

private:
};

TEST(ModelTests, WrongNumberOfOutputs) {
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
    EXPECT_NE(exc.givendim, coefficients.size());
    EXPECT_EQ(exc.dim, coefficients.size());
  }
}

class TestHessianImplementationModelWrongOutputDim : public Model {
public:

  inline TestHessianImplementationModelWrongOutputDim(pt::ptree const& pt) : Model(pt) {}

  virtual ~TestHessianImplementationModelWrongOutputDim() = default;
protected:

  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] coefficients The coefficients for each basis---this vector is devided into segments that correspond to coefficients of the bases. The length is the sum of the dimension of each basis.
  @param[in] bases The basis functions for each output
  \return The evaluation of \f$\mathcal{L}(u)\f$
  */
  inline virtual std::vector<Eigen::MatrixXd> OperatorHessianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    std::vector<Eigen::MatrixXd> hess(coefficients.size(), Eigen::MatrixXd(outputDimension, coefficients.size()));
    hess[0] = Eigen::MatrixXd(outputDimension+1, coefficients.size()+1);
    return hess;
  }

private:
};

TEST(ModelTests, WrongNumberOfOutputsHessianCheck) {
  // input/output dimension
  const std::size_t indim = 3, outdim = 5;

  // create a model
  pt::ptree pt;
  pt.put("InputDimension", indim);
  pt.put("OutputDimension", outdim);
  auto model = std::make_shared<TestHessianImplementationModelWrongOutputDim>(pt);

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
    EXPECT_NE(exc.givendim, outdim);
    EXPECT_EQ(exc.dim, outdim);
    EXPECT_NE(exc.givendimSecond, coefficients.size());
    EXPECT_EQ(exc.dimSecond, coefficients.size());
  }
}

class TestComponentWiseImplementationModel : public Model {
public:

  inline TestComponentWiseImplementationModel(pt::ptree const& pt) : Model(pt) {}

  virtual ~TestComponentWiseImplementationModel() = default;

  /// Compute the true Hessian of the operator with respect to the coefficients
  inline std::vector<Eigen::MatrixXd> TrueHessian(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
    std::vector<Eigen::MatrixXd> hess(coefficients.size());
    for( std::size_t i=0; i<hess.size(); ++i ) { hess[i] = Eigen::MatrixXd::Zero(outputDimension, coefficients.size()); }

    std::size_t ind = 0;
    for( std::size_t i=0; i<outputDimension; ++i ) {
      Eigen::VectorXd phi = bases[i]->EvaluateBasisFunctions(x);
      for( std::size_t k=0; k<phi.size(); ++k ) {
        for( std::size_t j=0; j<phi.size(); ++j ) {
          hess[ind+k](i, ind+j) = 2.0*phi(k)*phi(j);
        }
      }
      ind += bases[i]->NumBasisFunctions();
    }

    return hess;
  }

protected:

  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] outind Return this component of the evaluation of \f$f\f$
  \return The component of \f$f(x)\f$ corresponding to <tt>outind</tt>
  */
  inline virtual double RightHandSideComponentImpl(Eigen::VectorXd const& x, std::size_t const outind) const override { return x.prod(); }

  /// The operator \f$\mathcal{L}(u) = u^2\f$
  /**
  */
  inline virtual Eigen::VectorXd OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    assert(bases.size()==outputDimension);
    Eigen::VectorXd output = IdentityOperator(x, coefficients, bases);
    output = output.array()*output.array();

    return output;
  }

  inline virtual Eigen::MatrixXd OperatorJacobianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(outputDimension, coefficients.size());

    std::size_t ind = 0;
    for( std::size_t i=0; i<outputDimension; ++i ) {
      std::size_t basisSize = bases[i]->NumBasisFunctions();
      const Eigen::VectorXd phi = bases[i]->EvaluateBasisFunctions(x);
      assert(phi.size()==basisSize);
      jac.row(i).segment(ind, basisSize) = phi.dot(coefficients.segment(ind, basisSize))*phi;
      ind += basisSize;
    }

    return 2.0*jac;
  }

private:
};

TEST(ModelTests, ComponentWiseImplementation) {
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
  EXPECT_NEAR((jacFD-jac).norm(), 0.0, 1.0e-5);

  // evaluate the hessian of the operator
  const std::vector<Eigen::MatrixXd> trueHess = model->TrueHessian(x, coefficients, bases);
  const std::vector<Eigen::MatrixXd> hessFD = model->OperatorHessian(x, coefficients, bases);
  EXPECT_EQ(trueHess.size(), coefficients.size());
  EXPECT_EQ(trueHess.size(), hessFD.size());
  for( std::size_t i=0; i<trueHess.size(); ++i ) {
    EXPECT_EQ(trueHess[i].rows(), outdim);
    EXPECT_EQ(trueHess[i].cols(), coefficients.size());
    EXPECT_EQ(hessFD[i].rows(), outdim);
    EXPECT_EQ(hessFD[i].cols(), coefficients.size());
    EXPECT_NEAR((trueHess[i]-hessFD[i]).norm(), 0.0, 1.0e-8);
  }
}

class TestHessianImplementationModel : public Model {
public:

  inline TestHessianImplementationModel(pt::ptree const& pt) : Model(pt) {}

  virtual ~TestHessianImplementationModel() = default;

protected:

  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] outind Return this component of the evaluation of \f$f\f$
  \return The component of \f$f(x)\f$ corresponding to <tt>outind</tt>
  */
  inline virtual double RightHandSideComponentImpl(Eigen::VectorXd const& x, std::size_t const outind) const override { return x.prod(); }

  /// The operator \f$\mathcal{L}(u) = u^2\f$
  /**
  */
  inline virtual Eigen::VectorXd OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    assert(bases.size()==outputDimension);
    Eigen::VectorXd output = IdentityOperator(x, coefficients, bases);
    output = output.array()*output.array();

    return output;
  }

  inline virtual Eigen::MatrixXd OperatorJacobianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override {
    Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(outputDimension, coefficients.size());

    std::size_t ind = 0;
    for( std::size_t i=0; i<outputDimension; ++i ) {
      std::size_t basisSize = bases[i]->NumBasisFunctions();
      const Eigen::VectorXd phi = bases[i]->EvaluateBasisFunctions(x);
      assert(phi.size()==basisSize);
      jac.row(i).segment(ind, basisSize) = phi.dot(coefficients.segment(ind, basisSize))*phi;
      ind += basisSize;
    }

    return 2.0*jac;
  }

  /// Compute the true Hessian of the operator with respect to the coefficients
  inline std::vector<Eigen::MatrixXd> OperatorHessianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
    std::vector<Eigen::MatrixXd> hess(coefficients.size());
    for( std::size_t i=0; i<hess.size(); ++i ) { hess[i] = Eigen::MatrixXd::Zero(outputDimension, coefficients.size()); }

    std::size_t ind = 0;
    for( std::size_t i=0; i<outputDimension; ++i ) {
      Eigen::VectorXd phi = bases[i]->EvaluateBasisFunctions(x);
      for( std::size_t k=0; k<phi.size(); ++k ) {
        for( std::size_t j=0; j<phi.size(); ++j ) {
          hess[ind+k](i, ind+j) = 2.0*phi(k)*phi(j);
        }
      }
      ind += bases[i]->NumBasisFunctions();
    }

    return hess;
  }

private:
};

TEST(ModelTests, HessianImplementation) {
  // input/output dimension
  const std::size_t indim = 3, outdim = 5;

  // create a model
  pt::ptree pt;
  pt.put("InputDimension", indim);
  pt.put("OutputDimension", outdim);
  auto model = std::make_shared<TestHessianImplementationModel>(pt);

  // check in the input/output sizes
  EXPECT_EQ(model->inputDimension, indim);
  EXPECT_EQ(model->outputDimension, outdim);

  // pick a random point
  const Eigen::VectorXd x = Eigen::VectorXd::Random(indim);

  // try to evaluate the right hand side
  const Eigen::VectorXd rhs = model->RightHandSide(x);
  EXPECT_EQ(rhs.size(), outdim);
  for( std::size_t i=0; i<outdim; ++i ) { EXPECT_NEAR(rhs(i), x.prod(), 1.0e-12); }

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
  EXPECT_NEAR((jacFD-jac).norm(), 0.0, 1.0e-5);

  // evaluate the hessian of the operator
  const std::vector<Eigen::MatrixXd> hess = model->OperatorHessian(x, coefficients, bases);
  const std::vector<Eigen::MatrixXd> hessFD = model->OperatorHessianByFD(x, coefficients, bases);
  EXPECT_EQ(hess.size(), coefficients.size());
  EXPECT_EQ(hess.size(), hessFD.size());
  for( std::size_t i=0; i<hess.size(); ++i ) {
    EXPECT_EQ(hess[i].rows(), outdim);
    EXPECT_EQ(hess[i].cols(), coefficients.size());
    EXPECT_EQ(hessFD[i].rows(), outdim);
    EXPECT_EQ(hessFD[i].cols(), coefficients.size());
    EXPECT_NEAR((hess[i]-hessFD[i]).norm(), 0.0, 1.0e-8);
  }
}
