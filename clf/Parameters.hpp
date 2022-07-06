#ifndef PARAMETERS_HPP_
#define PARAMETERS_HPP_

#include <unordered_map>
#include <string>
#include <any>
#include <assert.h>

namespace clf {
  
 /// A class that holds algorithm parameters
class Parameters {
public:

  Parameters();

  virtual ~Parameters() = default;

  /// The number of stored parameters 
  /**
     \return The number of parameters in the map
   */
  std::size_t NumParameters() const;

  /// Add a parameter 
  /**
     @param[in] name The name of the parameters 
     @param[in] in The parameter's value
   */
  template<typename TYPE>
  inline void Add(std::string const& name, TYPE const& in) { map[name] = in; }

  /// Get a parameter
  /**
     @param[in] name The name of the parameters 
     \return The parameter's value
   */
  template<typename TYPE>
  inline TYPE Get(std::string const& name) const { 
    auto it = map.find(name);
    assert(it!=map.end());
    return std::any_cast<TYPE>(it->second);
  }

private:

  /// A map from the parameter name to its value (can be any type)
  std::unordered_map<std::string, std::any> map;
};

} // namespace clf

#endif
