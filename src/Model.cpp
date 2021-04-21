#include "clf/Model.hpp"

namespace pt = boost::property_tree;
using namespace clf;

Model::Model(pt::ptree const& pt) :
inputDimension(pt.get<std::size_t>("InputDimension", 1)),
outputDimension(pt.get<std::size_t>("OutputDimension", 1)),
fdEps(pt.get<double>("FiniteDifferenceStep", 1.0e-6))
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
   if( x.size()!=inputDimension) { throw exceptions::ModelHasWrongInputOutputDimensions(exceptions::ModelHasWrongInputOutputDimensions::Type::INPUT, exceptions::ModelHasWrongInputOutputDimensions::Function::OPERATOR_JACOBIAN, x.size(), inputDimension); }

   // the jacobian of the operator at x
   Eigen::MatrixXd jac;

   // try to evaluate the model jacobian
   try {
     jac = OperatorJacobianImpl(x, coefficients, bases);
   } catch( exceptions::ModelHasNotImplemented const& exc ) {
     // try to evaluate the model using impl
     Eigen::VectorXd u;
     try {
       u = OperatorImpl(x, coefficients, bases);
     } catch( exceptions::ModelHasNotImplemented const& exc ) {
       // the operator HAS NOT been implemented---return the jacobian of the identity
       jac = IdentityOperatorJacboian(x, bases, coefficients.size());
     }

     // the operator HAS been implemented (but not the Jacobian) so return the finite difference approximation
     if( jac.size()==0 ) { jac = OperatorJacobianByFD(x, coefficients, bases); }
   }

   if( jac.rows()!=outputDimension | jac.cols()!=coefficients.size() ) { throw exceptions::ModelHasWrongInputOutputDimensions(exceptions::ModelHasWrongInputOutputDimensions::Function::OPERATOR_JACOBIAN, jac.rows(), outputDimension, jac.cols(), coefficients.size()); }

   return jac;
 }

 Eigen::MatrixXd Model::OperatorJacobianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
   throw exceptions::ModelHasNotImplemented(exceptions::ModelHasNotImplemented::Type::BOTH, exceptions::ModelHasNotImplemented::Function::OPERATOR_JACOBIAN);
   return Eigen::MatrixXd();
 }

 Eigen::MatrixXd Model::OperatorJacobianByFD(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
    // compute the reference action of the operator
    const Eigen::VectorXd u = Operator(x, coefficients, bases);

    // the operator jacobian with respect to the coefficients
    Eigen::MatrixXd jac(outputDimension, coefficients.size());

    // the coefficients plus epsilon
    Eigen::VectorXd coeffPlus = coefficients;
    for( std::size_t i=0; i<coefficients.size(); ++i ) {
      // add to the ith coefficient
      if( i>0 ) { coeffPlus(i-1) -= fdEps; }
      coeffPlus(i) += fdEps;

      // compute the derivative using finite difference
      jac.col(i) = (Operator(x, coeffPlus, bases)-u)/fdEps;
    }

   return jac;
 }

 Eigen::MatrixXd Model::IdentityOperatorJacboian(Eigen::VectorXd const& x, std::vector<std::shared_ptr<const BasisFunctions> > const& bases, std::size_t numCoeffs) const {
   if( numCoeffs==0 ) { for( const auto& basis : bases ) { numCoeffs += basis->NumBasisFunctions(); } }

   Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(outputDimension, numCoeffs);

   // the running starting index
   std::size_t ind = 0;
   for( std::size_t i=0; i<bases.size(); ++i ) {
     const std::size_t basisSize = bases[i]->NumBasisFunctions();
     jac.row(i).segment(ind, basisSize) = bases[i]->EvaluateBasisFunctions(x);
     ind += basisSize;
   }
   assert(ind==numCoeffs);

   return jac;
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
