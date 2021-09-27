#ifndef SHAREDFACTORY_HPP_
#define SHAREDFACTORY_HPP_

#include <memory>

// this file is based on a snippet: http://pastebin.com/dSTLt7vW; also very similar to the trick used in the MUQ::Utilities library

namespace clf {


/// A structure that creates constructors given a string 
/**
Contains a basic prescription for performing string registration. Allows users to use a string to get a function pointer to a constructor.
*/
template<typename T>
struct SharedFactory {
  /// The operator that maps from the constructor arguments for type T and returns a pointer to a new instance of T
  template<typename... Args> 
  std::shared_ptr<T> operator()(Args... args) { return std::make_shared<T>(args...); }
};

} // namespace clf


#endif
