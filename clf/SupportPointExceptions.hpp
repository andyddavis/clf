#ifndef SUPPORTPOINTEXCEPTIONS_HPP_
#define SUPPORTPOINTEXCEPTIONS_HPP_

#include "clf/CLFException.hpp"

namespace clf {

/// The support point tried to create its basis functions in an invalid way
class SupportPointBasisException : virtual public CLFException {
public:

  /**
  @param[in] basisType The (invalid) basis that the support point tried to create
  */
  SupportPointBasisException(std::string const& basisType);

  virtual ~SupportPointBasisException() = default;

  /// The (invalid) basis that the support point tried to create
  const std::string basisType;
  
private:

};

} // namespace clf

#endif
