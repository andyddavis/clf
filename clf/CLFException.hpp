#ifndef CLF_EXCEPTION_HPP_
#define CLF_EXCEPTION_HPP_

#include <exception>
#include <string>

namespace clf {
namespace exceptions {

/// A generic exception for the coupled local function library
class CLFException  : virtual public std::exception {
public:

  /// The default constructor
  CLFException();

  virtual ~CLFException() = default;

  /// The error message that gets printed when this exception is thrown
  virtual const char* what() const noexcept;

protected:

  /// The printed error message when this exception is thrown
  std::string message = "CLF ERROR: UNKNOWN";

private:
};

} // namespace exceptions
} // namespace clf

#endif
