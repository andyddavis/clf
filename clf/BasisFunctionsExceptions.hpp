#ifndef BASISFUNCTIONSEXCEPTIONS_HPP_
#define BASISFUNCTIONSEXCEPTIONS_HPP_

#include <exception>
#include <string>

namespace clf {

/// Make sure the basis that we tried to construct is a valid basis option
class BasisFunctionsNameConstuctionException : virtual public std::exception {
public:

  /**
  @param[in] basisName The basis name that we tried (and failed) to construct
  */
  BasisFunctionsNameConstuctionException(std::string const& basisName);

  virtual ~BasisFunctionsNameConstuctionException() = default;

  /// The error message that gets printed when this exception is thrown
  virtual const char* what() const noexcept;

  /// The basis name that we tried (and failed) to construct
  const std::string basisName;

private:

  /// The printed error message when this exception is thrown
  std::string message = "UNKNOWN ERROR";
};

} // namespace clf

#endif
