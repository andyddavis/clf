#ifndef BASISFUNCTIONSEXCEPTIONS_HPP_
#define BASISFUNCTIONSEXCEPTIONS_HPP_

#include "clf/CLFException.hpp"

namespace clf {
namespace exceptions {

/// Make sure the basis that we tried to construct is a valid basis option
class BasisFunctionsNameConstuctionException : virtual public CLFException {
public:

  /**
  @param[in] basisName The basis name that we tried (and failed) to construct
  */
  BasisFunctionsNameConstuctionException(std::string const& basisName);

  virtual ~BasisFunctionsNameConstuctionException() = default;

  /// The basis name that we tried (and failed) to construct
  const std::string basisName;

private:
};

} // namespace exceptions
} // namespace clf

#endif
