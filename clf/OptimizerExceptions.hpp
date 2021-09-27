#ifndef OPTIMIZEREXCEPTIONS_HPP_
#define OPTIMIZEREXCEPTIONS_HPP_

#include "clf/CLFException.hpp"

namespace clf {
namespace exceptions {

/// Make sure the type of optimizer we are trying to construct is valid 
template<typename OptimizerType>
class OptimizerNameException : virtual public CLFException {
public:

  /// Construct the exception that is thrown if we cannot construct this optimizer
  /**
  @param[in] name The name of the optimizer we tried and failed to construct
  */
  inline OptimizerNameException(std::string const& name) :
  name(name) 
  {
    message = "ERROR: Could not find optimizer with name: " + name + ". Options are:\n";
    auto map = OptimizerType::GetOptimizerConstructorMap();
    for( auto it=map->begin(); it!=map->end(); ++it ) { message += it->first + "\n"; }
  }

  virtual ~OptimizerNameException() = default;

  /// The name of the optimizer we tried and failed to construct
  const std::string name;
private:
};

} // namespace exceptions
} // namespace clf 


#endif
