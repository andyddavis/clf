#ifndef MODELEXCEPTIONS_HPP_
#define MODELEXCEPTIONS_HPP_

#include <cassert>

#include <vector>

#include "clf/CLFException.hpp"

namespace clf {
namespace exceptions {

/// The model was given the wrong input size or the computed output is the wrong size
class ModelHasWrongInputOutputDimensions : virtual public CLFException {
public:
  /// Is the input or output dimension wrong?
  enum Type {
    /// The input dimension is wrong
    INPUT,

    /// The output dimension is wrong
    OUTPUT
  };

  /// Was the right hand side or the operator called with the wrong dimension?
  enum Function {
    /// The right hand side is wrong
    RHS,

    /// The operator is wrong
    OPERATOR,

    /// The Jacobian of the operator with respect to the coefficients is wrong
    OPERATOR_JACOBIAN
  };

  /**
  @param[in] type Was the wrong dimension in the input or output?
  @param[in] func Was the wrong dimension when calling the right hand side function or the operator?
  @param[in] givendim The given dimension
  @param[in] dim The expected dimension
  */
  ModelHasWrongInputOutputDimensions(Type const& type, Function const& func, std::size_t const givendim, std::size_t const dim);

  /**
  @param[in] func Was the wrong dimension when calling the right hand side function or the operator?
  @param[in] givendim1 The given first dimension
  @param[in] dim1 The expected first dimension
  @param[in] givendim2 The given second dimension
  @param[in] dim2 The expected second dimension
  */
  ModelHasWrongInputOutputDimensions(Function const& func, std::size_t const givendim1, std::size_t const dim1, std::size_t const givendim2, std::size_t const dim2);

  virtual ~ModelHasWrongInputOutputDimensions() = default;

  /// Was the wrong dimension in the input or output?
  const Type type;

  /// Was the wrong dimension when calling the right hand side function or the operator?
  const Function func;

  /// The given dimension
  const std::size_t givendim;

  /// The expected dimension
  const std::size_t dim;

  /// The given second dimension
  /**
  Used to check matrix sizes, defaults to 1.
  */
  const std::size_t givendimSecond = 1;

  /// The expected second dimension
  /**
  Used to check matrix sizes, defaults to 1.
  */
  const std::size_t dimSecond = 1;

private:

  /// The dimensions for the right hand side are wrong
  void WrongRHSDimensions();

  /// The dimensions for the operator are wrong
  void WrongOperatorDimensions();

  /// The dimensions for the Jacobian of the operator are wrong
  /**
  @param[in] type Is the input or output dimension wrong?
  */
  void WrongOperatorJacobianDimensions(Type const& type);
};

/// The model has not implemented the right hand side function or the operator
class ModelHasNotImplemented : virtual public CLFException {
public:

  /// Which potential implementation threw this exception?
  enum Type {
    /// Vector implementation
    VECTOR,

    /// Component implementation
    COMPONENT,

    /// Neither vector nor componet-wise implementation has been detected
    BOTH
  };

  /// Was the right hand side or the operator called not implemented
  enum Function {
    /// The right hand side is not implemented
    RHS,

    /// The operator is not implemented
    OPERATOR,

    /// The Jacobian of the operator is not implemented
    OPERATOR_JACOBIAN
  };


  ModelHasNotImplemented(Type const& type, Function const& func);

  virtual ~ModelHasNotImplemented() = default;

  /// Which potential implementation threw this exception?
  const Type type;

  /// Was the right hand side function or the operator not implemented?
  const Function func;
private:
};

} // namespace exceptions
} // namespace exceptions

#endif
