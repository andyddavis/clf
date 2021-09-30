#include "clf/ModelExceptions.hpp"

using namespace clf::exceptions;

ModelHasWrongInputOutputDimensions::ModelHasWrongInputOutputDimensions(Function const& func, std::size_t const givendim1, std::size_t const dim1, std::size_t const givendim2, std::size_t const dim2) : CLFException(), type(Type::OUTPUT), func(func), givendim(givendim1), dim(dim1), givendimSecond(givendim2), dimSecond(dim2) {
  assert(func!=Function::RHS); assert(func!=Function::OPERATOR);

  if( func==Function::OPERATOR_JACOBIAN ) {
    WrongOperatorJacobianDimensions(Type::OUTPUT);
  } else if( func==Function::OPERATOR_HESSIAN_MATRIX ) {
    assert(type==Type::OUTPUT);
    WrongOperatorHessianDimensions(Type::OUTPUT, Function::OPERATOR_HESSIAN_MATRIX);
  }
}

ModelHasWrongInputOutputDimensions::ModelHasWrongInputOutputDimensions(Type const& type, Function const& func, std::size_t const givendim, std::size_t const dim) : CLFException(), type(type), func(func), givendim(givendim), dim(dim) {
  if( func==Function::RHS ) {
    WrongRHSDimensions();
  } else if( func==Function::OPERATOR ) {
    WrongOperatorDimensions();
  } else if( func==Function::OPERATOR_JACOBIAN ) {
    assert(type==Type::INPUT);
    WrongOperatorJacobianDimensions(Type::INPUT);
  } else if( func==Function::OPERATOR_HESSIAN ) {
    assert(type==Type::INPUT);
    WrongOperatorHessianDimensions(Type::INPUT, Function::OPERATOR_HESSIAN);
  } else if( func==Function::OPERATOR_HESSIAN_VECTOR ) {
    assert(type==Type::OUTPUT);
    WrongOperatorHessianDimensions(Type::OUTPUT, Function::OPERATOR_HESSIAN_VECTOR);
  } else if( func==Function::FUNCTION_DERIVATIVE ) {
    assert(type==Type::INPUT);
    WrongFunctionDerivativeDimensions();
  }
}

void ModelHasWrongInputOutputDimensions::WrongOperatorHessianDimensions(Type const& type, Function const& func) {
  if( type==Type::INPUT ) {
    if( dim==1 ) {
      message = "ERROR: clf::Model::OperatorHessian was called with a " + std::to_string(givendim) + "-dimensional vector but requires " + std::to_string(dim) + " dimension.";
    } else {
      message = "ERROR: clf::Model::OperatorHessian was called with a " + std::to_string(givendim) + "-dimensional vector but requires " + std::to_string(dim) + " dimensions.";
    }
  } else {
    if( func==Function::OPERATOR_HESSIAN_VECTOR ) {
      message = "ERROR: clf::Model::OperatorHessian returned a " + std::to_string(givendim) + "-dimensional vector of matrices but required " + std::to_string(dim) + "-dimensional vector of matrices.";
    } else {
      message = "ERROR: clf::Model::OperatorHessian returned a " + std::to_string(givendim) + "x" + std::to_string(givendimSecond) + " matrix (in the vector of matrices) but required " + std::to_string(dim) + "x" + std::to_string(dimSecond) + " matrix.";
    }
  }
}

void ModelHasWrongInputOutputDimensions::WrongOperatorJacobianDimensions(Type const& type) {
  if( type==Type::INPUT ) {
    if( dim==1 ) {
      message = "ERROR: clf::Model::OperatorJacobian was called with a " + std::to_string(givendim) + "-dimensional vector but requires " + std::to_string(dim) + " dimension.";
    } else {
      message = "ERROR: clf::Model::OperatorJacobian was called with a " + std::to_string(givendim) + "-dimensional vector but requires " + std::to_string(dim) + " dimensions.";
    }
  } else {
    message = "ERROR: clf::Model::OperatorJacobian returned a " + std::to_string(givendim) + "x" + std::to_string(givendimSecond) + " matrix but required " + std::to_string(dim) + "x" + std::to_string(dimSecond) + " matrix.";
  }
}

void ModelHasWrongInputOutputDimensions::WrongOperatorDimensions() {
  if( type==Type::INPUT ) {
    if( dim==1 ) {
      message = "ERROR: clf::Model::Operator was called with a " + std::to_string(givendim) + "-dimensional vector but requires " + std::to_string(dim) + " dimension.";
    } else {
      message = "ERROR: clf::Model::Operator was called with a " + std::to_string(givendim) + "-dimensional vector but requires " + std::to_string(dim) + " dimensions.";
    }
  } else {
    if( dim==1 ) {
      message = "ERROR: clf::Model::Operator returned a " + std::to_string(givendim) + "-dimensional vector but required " + std::to_string(dim) + " dimension.";
    } else {
      message = "ERROR: clf::Model::Operator returned a " + std::to_string(givendim) + "-dimensional vector but required " + std::to_string(dim) + " dimensions.";
    }
  }
}

void ModelHasWrongInputOutputDimensions::WrongRHSDimensions() {
  if( type==Type::INPUT ) {
    if( dim==1 ) {
      message = "ERROR: clf::Model::RightHandSide was called with a " + std::to_string(givendim) + "-dimensional vector but requires " + std::to_string(dim) + " dimension.";
    } else {
      message = "ERROR: clf::Model::RightHandSide was called with a " + std::to_string(givendim) + "-dimensional vector but requires " + std::to_string(dim) + " dimensions.";
    }
  } else {
    if( dim==1 ) {
      message = "ERROR: clf::Model::RightHandSide returned a " + std::to_string(givendim) + "-dimensional vector but required " + std::to_string(dim) + " dimension.";
    } else {
      message = "ERROR: clf::Model::RightHandSide returned a " + std::to_string(givendim) + "-dimensional vector but required " + std::to_string(dim) + " dimensions.";
    }
  }
}

void ModelHasWrongInputOutputDimensions::WrongFunctionDerivativeDimensions() {
  if( dim==1 ) {
    message = "ERROR: clf::Model::FunctionDerivative was called with a " + std::to_string(givendim) + "-dimensional vector but requires " + std::to_string(dim) + " dimension.";
  } else {
    message = "ERROR: clf::Model::FunctionDerivative was called with a " + std::to_string(givendim) + "-dimensional vector but requires " + std::to_string(dim) + " dimensions.";
  }
}

ModelHasNotImplemented::ModelHasNotImplemented(Type const& type, Function const& func) : CLFException(), type(type), func(func) {
  if( func==Function::RHS ) {
    message = "ERROR: clf::Model has not implemented the right hand side function.";
  } else if( func==Function::OPERATOR ) {
    message = "ERROR: clf::Model has not implemented the operator.";
  } else if( func==Function::OPERATOR_JACOBIAN ) {
    message = "ERROR: clf::Model has not implemented the Jacobian of the operator.";
  } else if( func==Function::OPERATOR_HESSIAN ) {
    message = "ERROR: clf::Model has not implemented the Hessian of the operator.";
  } else if( func==Function::LINEAR_MODEL_MATRIX ) {
    message = "ERROR: clf::Model has not implemented the matrix that defines a clf::LinearModel.";
  } else {
    message = "ERROR: clf::Model has not implemented something.";
  }

  if( type==Type::VECTOR ) {
    message += " Error generated by vector-valued implementation call";
  } else if( type==Type::COMPONENT ) {
    message += " Error generated by component-wise implementation call";
  }
  message += " (clf::exceptions::ModelHasNotImplemented).";
}
