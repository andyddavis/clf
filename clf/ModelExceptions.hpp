#ifndef MODELEXCEPTIONS_HPP_
#define MODELEXCEPTIONS_HPP_

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
    OPERATOR
  };

  /**
  @param[in] type Was the wrong dimension in the input or output?
  @param[in] func Was the wrong dimension when calling the right hand side function or the operator?
  @param[in] givendim The given dimension
  @param[in] dim The expected dimension
  */
  ModelHasWrongInputOutputDimensions(Type const& type, Function const& func, std::size_t const givendim, std::size_t const dim);

  virtual ~ModelHasWrongInputOutputDimensions() = default;

  /// Was the wrong dimension in the input or output?
  const Type type;

  /// Was the wrong dimension when calling the right hand side function or the operator?
  const Function func;

  /// The given dimension
  const std::size_t givendim;

  /// The expected dimension
  const std::size_t dim;

private:
};

/// The model has not implemented the right hand side function
class ModelHasNotImplementedRHS : virtual public CLFException {
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

  ModelHasNotImplementedRHS(Type const& type);

  virtual ~ModelHasNotImplementedRHS() = default;

  /// Which potential implementation threw this exception?
  Type type;
private:
};

} // namespace exceptions
} // namespace exceptions

#endif
