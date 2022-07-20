#ifndef CLFEXCEPTIONS_HPP_
#define CLFEXCEPTIONS_HPP_

#include <stdexcept>
#include <string>

namespace clf {
namespace exceptions {

/// Something is not implemented
class NotImplemented : public std::logic_error {
public:
  /**
     @param[in] name The name of the thing that is not implemented
   */
  NotImplemented(std::string const& name);

  virtual ~NotImplemented() = default;
private:
};

} // namespace exceptions
} // namespace clf

#endif
