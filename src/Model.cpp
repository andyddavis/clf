#include "clf/Model.hpp"

namespace pt = boost::property_tree;
using namespace clf;

Model::Model(pt::ptree const& pt) :
inputDimension(pt.get<std::size_t>("InputDimension", 1)),
outputDimension(pt.get<std::size_t>("OutputDimension", 1))
{}

double Model::NearestNeighborKernel(double const delta) const {
  assert(delta>-1.0e-10);
  if( delta>1.0+1.0e-10 ) { return 0.0; }
  return 1.0;
}

Eigen::VectorXd Model::Operator(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
  if( x.size()!=inputDimension) { throw exceptions::ModelHasWrongInputOutputDimensions(exceptions::ModelHasWrongInputOutputDimensions::Type::INPUT, exceptions::ModelHasWrongInputOutputDimensions::Function::OPERATOR, x.size(), inputDimension); }

  // the action of the operator at x
  Eigen::VectorXd output;

  // try to call the implementation of the operator
  try {
    output = OperatorImpl(x, coefficients, bases);
  } catch( exceptions::ModelHasNotImplemented const& exc ) {
    // the user has not implemented the operator, so use the identity
    output = IdentityOperator(x, coefficients, bases);
  }

  // make sure the output dimension is correct
  if( output.size()!=outputDimension) { throw exceptions::ModelHasWrongInputOutputDimensions(exceptions::ModelHasWrongInputOutputDimensions::Type::OUTPUT, exceptions::ModelHasWrongInputOutputDimensions::Function::OPERATOR, output.size(), outputDimension); }

  return output;
}

Eigen::VectorXd Model::IdentityOperator(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
  if( x.size()!=inputDimension) { throw exceptions::ModelHasWrongInputOutputDimensions(exceptions::ModelHasWrongInputOutputDimensions::Type::INPUT, exceptions::ModelHasWrongInputOutputDimensions::Function::OPERATOR, x.size(), inputDimension); }

  assert(bases.size()==outputDimension);
  Eigen::VectorXd output(outputDimension);
  std::size_t runningind = 0;
  for( std::size_t i=0; i<outputDimension; ++i ) {
    assert(runningind+bases[i]->NumBasisFunctions()<=coefficients.size());
    output(i) = bases[i]->FunctionEvaluation(x, coefficients.segment(runningind, bases[i]->NumBasisFunctions()));
    runningind += bases[i]->NumBasisFunctions();
  }
  return output;
}

 Eigen::VectorXd Model::OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
   throw exceptions::ModelHasNotImplemented(exceptions::ModelHasNotImplemented::Type::BOTH, exceptions::ModelHasNotImplemented::Function::OPERATOR);
   return Eigen::VectorXd();
 }

 Eigen::MatrixXd Model::OperatorJacobian(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
   return Eigen::MatrixXd();
 }

Eigen::VectorXd Model::RightHandSide(Eigen::VectorXd const& x) const {
  if( x.size()!=inputDimension) { throw exceptions::ModelHasWrongInputOutputDimensions(exceptions::ModelHasWrongInputOutputDimensions::Type::INPUT, exceptions::ModelHasWrongInputOutputDimensions::Function::RHS, x.size(), inputDimension); }
  Eigen::VectorXd rhs;

  // try to call the vector implementation
  try {
    rhs = RightHandSideVectorImpl(x);
  } catch( exceptions::ModelHasNotImplemented const& excvector ) {
    try {
      rhs.resize(outputDimension);
      for( std::size_t i=0; i<outputDimension; ++i ) { rhs(i) = RightHandSideComponentImpl(x, i); }
    } catch( exceptions::ModelHasNotImplemented const& exccomponent ) {
      throw exceptions::ModelHasNotImplemented(exceptions::ModelHasNotImplemented::Type::BOTH, exceptions::ModelHasNotImplemented::Function::RHS);
    }
  }

  if( rhs.size()!=outputDimension) { throw exceptions::ModelHasWrongInputOutputDimensions(exceptions::ModelHasWrongInputOutputDimensions::Type::OUTPUT, exceptions::ModelHasWrongInputOutputDimensions::Function::RHS, rhs.size(), outputDimension); }

  return rhs;
}

Eigen::VectorXd Model::RightHandSideVectorImpl(Eigen::VectorXd const& x) const {
  throw exceptions::ModelHasNotImplemented(exceptions::ModelHasNotImplemented::Type::VECTOR, exceptions::ModelHasNotImplemented::Function::RHS);
  return Eigen::VectorXd();
}

double Model::RightHandSideComponentImpl(Eigen::VectorXd const& x, std::size_t const outind) const {
  throw exceptions::ModelHasNotImplemented(exceptions::ModelHasNotImplemented::Type::COMPONENT, exceptions::ModelHasNotImplemented::Function::RHS);
  return 0.0;
}
