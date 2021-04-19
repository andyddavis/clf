#ifndef MODELEXCEPTIONS_HPP_
#define MODELEXCEPTIONS_HPP_

#include <vector>

#include "clf/CLFException.hpp"

namespace clf {
namespace exceptions {

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
