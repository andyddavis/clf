#include "clf/Model.hpp"

namespace pt = boost::property_tree;
using namespace clf;

Model::Model(std::size_t const indim, std::size_t const outdim) :
inputDimension(indim),
outputDimension(outdim)
{}

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
  output = OperatorImpl(x, coefficients, bases);

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

Eigen::MatrixXd Model::OperatorJacobian(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
   if( x.size()!=inputDimension) { throw exceptions::ModelHasWrongInputOutputDimensions(exceptions::ModelHasWrongInputOutputDimensions::Type::INPUT, exceptions::ModelHasWrongInputOutputDimensions::Function::OPERATOR_JACOBIAN, x.size(), inputDimension); }

   // the jacobian of the operator at x
   Eigen::MatrixXd jac;

   // try to evaluate the model jacobian
   try {
     jac = OperatorJacobianImpl(x, coefficients, bases);
   } catch( exceptions::ModelHasNotImplemented const& exc ) {
     jac = OperatorJacobianByFD(x, coefficients, bases); 
   }

   if( jac.rows()!=outputDimension | jac.cols()!=coefficients.size() ) { throw exceptions::ModelHasWrongInputOutputDimensions(exceptions::ModelHasWrongInputOutputDimensions::Function::OPERATOR_JACOBIAN, jac.rows(), outputDimension, jac.cols(), coefficients.size()); }

   return jac;
 }

 Eigen::MatrixXd Model::OperatorJacobianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
   throw exceptions::ModelHasNotImplemented(exceptions::ModelHasNotImplemented::Type::BOTH, exceptions::ModelHasNotImplemented::Function::OPERATOR_JACOBIAN);
   return Eigen::MatrixXd();
 }

Eigen::MatrixXd Model::OperatorJacobianByFD(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases, FDOrder const order, Eigen::VectorXd const& eval, double const fdEps) const {
    // compute the reference action of the operator
    Eigen::VectorXd u;
    if( order==FDOrder::FIRST_UPWARD | order==FDOrder::FIRST_DOWNWARD ) { 
      if( eval.size()==0 ) { u = Operator(x, coefficients, bases); }
      assert(eval.size()==0 || u.size()==0);
      assert(eval.size()==outputDimension || u.size()==outputDimension);
    }

    // the operator jacobian with respect to the coefficients
    Eigen::MatrixXd jac(outputDimension, coefficients.size());

    // the coefficients plus epsilon
    Eigen::VectorXd coeffFD = coefficients;
    for( std::size_t i=0; i<coefficients.size(); ++i ) {
      switch( order ) {
      case FDOrder::FIRST_UPWARD: {
	// compute the derivative using finite difference
	coeffFD(i) += fdEps;
	jac.col(i) = (Operator(x, coeffFD, bases)-(eval.size()==0? u : eval))/fdEps;
	coeffFD(i) -= fdEps;
	break;
      }
      case FDOrder::FIRST_DOWNWARD: {
	// compute the derivative using finite difference
	coeffFD(i) -= fdEps;
	jac.col(i) = ((eval.size()==0? u : eval)-Operator(x, coeffFD, bases))/fdEps;
	coeffFD(i) += fdEps;
	break;
      }
      case FDOrder::SECOND: {
	// compute the derivative using finite difference
	coeffFD(i) -= fdEps;
	const Eigen::VectorXd m = Operator(x, coeffFD, bases);
	coeffFD(i) += 2.0*fdEps;
	const Eigen::VectorXd p = Operator(x, coeffFD, bases);
	coeffFD(i) -= fdEps;
	jac.col(i) = (p-m)/(2.0*fdEps);
	break;
      }
      case FDOrder::FOURTH: {
	// compute the derivative using finite difference
	coeffFD(i) -= fdEps;
	const Eigen::VectorXd m1 = Operator(x, coeffFD, bases);
	coeffFD(i) -= fdEps;
	const Eigen::VectorXd m2 = Operator(x, coeffFD, bases);
	coeffFD(i) += 3.0*fdEps;
	const Eigen::VectorXd p1 = Operator(x, coeffFD, bases);
	coeffFD(i) += fdEps;
	const Eigen::VectorXd p2 = Operator(x, coeffFD, bases);
	coeffFD(i) -= 2.0*fdEps;
	jac.col(i) = ((m2-p2)/12.0+(2.0/3.0)*(p1-m1))/fdEps;
	break;
      }
      case FDOrder::SIXTH: {
	// compute the derivative using finite difference
	coeffFD(i) -= fdEps;
	const Eigen::VectorXd m1 = Operator(x, coeffFD, bases);
	coeffFD(i) -= fdEps;
	const Eigen::VectorXd m2 = Operator(x, coeffFD, bases);
	coeffFD(i) -= fdEps;
	const Eigen::VectorXd m3 = Operator(x, coeffFD, bases);
	coeffFD(i) += 4.0*fdEps;
	const Eigen::VectorXd p1 = Operator(x, coeffFD, bases);
	coeffFD(i) += fdEps;
	const Eigen::VectorXd p2 = Operator(x, coeffFD, bases);
	coeffFD(i) += fdEps;
	const Eigen::VectorXd p3 = Operator(x, coeffFD, bases);
	coeffFD(i) -= 3.0*fdEps;
	jac.col(i) = ((p3-m3)/60.0+(3.0/20.0)*(m2-p2)+(3.0/4.0)*(p1-m1))/fdEps;
	break;
      }
      }
    }

   return jac;
 }

 Eigen::MatrixXd Model::IdentityOperatorJacobian(Eigen::VectorXd const& x, std::vector<std::shared_ptr<const BasisFunctions> > const& bases, std::size_t numCoeffs) const {
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

std::vector<Eigen::MatrixXd> Model::OperatorHessianByFD(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases, FDOrder const order, Eigen::MatrixXd const& jacEval, double const fdEps) const {
   // compute the reference jacobian
   Eigen::MatrixXd jac;
   if( order==FDOrder::FIRST_UPWARD | order==FDOrder::FIRST_DOWNWARD ) { 
     if( jacEval.size()==0 ) { jac = OperatorJacobian(x, coefficients, bases); }
     assert(jacEval.size()==0 || jac.size()==0);
     assert(jacEval.rows()==outputDimension || jac.rows()==outputDimension);
     assert(jacEval.cols()==coefficients.size() || jac.cols()==coefficients.size());
   }

   std::vector<Eigen::MatrixXd> hess(outputDimension, Eigen::MatrixXd(coefficients.size(), coefficients.size()));

   // the coefficients plus epsilon
    Eigen::VectorXd coeffFD = coefficients;
    for( std::size_t i=0; i<coefficients.size(); ++i ) {
      Eigen::MatrixXd secondDeriv;
      switch( order ) {
      case FDOrder::FIRST_UPWARD: {
	coeffFD(i) += fdEps;
	// compute the second derivative using finite difference
	secondDeriv = (OperatorJacobian(x, coeffFD, bases)-(jac.size()==0? jacEval : jac))/fdEps;
	coeffFD(i) -= fdEps;
	break;
      }
      case FDOrder::FIRST_DOWNWARD: {
	coeffFD(i) -= fdEps;
	// compute the second derivative using finite difference
	secondDeriv = ((jac.size()==0? jacEval : jac)-OperatorJacobian(x, coeffFD, bases))/fdEps;
	coeffFD(i) += fdEps;
	break;
      }
      case FDOrder::SECOND: {
	// compute the derivative using finite difference
	coeffFD(i) -= fdEps;
	const Eigen::MatrixXd m = OperatorJacobian(x, coeffFD, bases);
	coeffFD(i) += 2.0*fdEps;
	const Eigen::MatrixXd p = OperatorJacobian(x, coeffFD, bases);
	coeffFD(i) -= fdEps;
	secondDeriv = (p-m)/(2.0*fdEps);
	break;
      }
      case FDOrder::FOURTH: {
	// compute the derivative using finite difference
	coeffFD(i) -= fdEps;
	const Eigen::MatrixXd m1 = OperatorJacobian(x, coeffFD, bases);
	coeffFD(i) -= fdEps;
	const Eigen::MatrixXd m2 = OperatorJacobian(x, coeffFD, bases);
	coeffFD(i) += 3.0*fdEps;
	const Eigen::MatrixXd p1 = OperatorJacobian(x, coeffFD, bases);
	coeffFD(i) += fdEps;
	const Eigen::MatrixXd p2 = OperatorJacobian(x, coeffFD, bases);
	coeffFD(i) -= 2.0*fdEps;
	secondDeriv = ((m2-p2)/12.0+(2.0/3.0)*(p1-m1))/fdEps;
	break;
      }
      case FDOrder::SIXTH: {
	// compute the derivative using finite difference
	coeffFD(i) -= fdEps;
	const Eigen::MatrixXd m1 = OperatorJacobian(x, coeffFD, bases);
	coeffFD(i) -= fdEps;
	const Eigen::MatrixXd m2 = OperatorJacobian(x, coeffFD, bases);
	coeffFD(i) -= fdEps;
	const Eigen::MatrixXd m3 = OperatorJacobian(x, coeffFD, bases);
	coeffFD(i) += 4.0*fdEps;
	const Eigen::MatrixXd p1 = OperatorJacobian(x, coeffFD, bases);
	coeffFD(i) += fdEps;
	const Eigen::MatrixXd p2 = OperatorJacobian(x, coeffFD, bases);
	coeffFD(i) += fdEps;
	const Eigen::MatrixXd p3 = OperatorJacobian(x, coeffFD, bases);
	coeffFD(i) -= 3.0*fdEps;
	secondDeriv = ((p3-m3)/60.0+(3.0/20.0)*(m2-p2)+(3.0/4.0)*(p1-m1))/fdEps;
	break;
      }
      }
      for( std::size_t j=0; j<outputDimension; ++j ) { hess[j].row(i) = secondDeriv.row(j); }
    }

   return hess;
 }

 std::vector<Eigen::MatrixXd> Model::OperatorHessian(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
   if( x.size()!=inputDimension) { throw exceptions::ModelHasWrongInputOutputDimensions(exceptions::ModelHasWrongInputOutputDimensions::Type::INPUT, exceptions::ModelHasWrongInputOutputDimensions::Function::OPERATOR_HESSIAN, x.size(), inputDimension); }

   std::vector<Eigen::MatrixXd> hess;

   // try to evaluate the model hessian
   try {
     hess = OperatorHessianImpl(x, coefficients, bases);
   } catch( exceptions::ModelHasNotImplemented const& exc ) {
     hess = OperatorHessianByFD(x, coefficients, bases);
   }

   if( hess.size()!=outputDimension ) { throw exceptions::ModelHasWrongInputOutputDimensions(exceptions::ModelHasWrongInputOutputDimensions::Type::OUTPUT, exceptions::ModelHasWrongInputOutputDimensions::Function::OPERATOR_HESSIAN_VECTOR, hess.size(), outputDimension); }
   for( const auto& it : hess ) {
     if( it.rows()!=coefficients.size() | it.cols()!=coefficients.size() ) { throw exceptions::ModelHasWrongInputOutputDimensions(exceptions::ModelHasWrongInputOutputDimensions::Function::OPERATOR_HESSIAN_MATRIX, it.rows(), coefficients.size(), it.cols(), coefficients.size()); }
   }

   return hess;
 }

 std::vector<Eigen::MatrixXd> Model::OperatorHessianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
   throw exceptions::ModelHasNotImplemented(exceptions::ModelHasNotImplemented::Type::BOTH, exceptions::ModelHasNotImplemented::Function::OPERATOR_HESSIAN);
   return std::vector<Eigen::MatrixXd>();
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

double Model::FunctionDerivative(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases, std::size_t const i, std::size_t const j, std::size_t const k) const {
  assert(i<outputDimension); assert(j<inputDimension);
  assert(bases.size()==outputDimension);

  if( x.size()!=inputDimension) { throw exceptions::ModelHasWrongInputOutputDimensions(exceptions::ModelHasWrongInputOutputDimensions::Type::INPUT, exceptions::ModelHasWrongInputOutputDimensions::Function::FUNCTION_DERIVATIVE, x.size(), inputDimension); }

  std::size_t firstind = 0;
  for( std::size_t ind=0; ind<i; ++ind ) { firstind += bases[ind]->NumBasisFunctions(); }

  return bases[i]->FunctionDerivative(x, coefficients.segment(firstind, bases[i]->NumBasisFunctions()), j, k);
}

bool Model::IsLinear() const { return false; }
