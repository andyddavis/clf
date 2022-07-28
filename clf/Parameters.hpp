#ifndef PARAMETERS_HPP_
#define PARAMETERS_HPP_

#include <unordered_map>
#include <string>
#include <any>
#include <assert.h>
#include <iostream>

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
    if( it==map.end() ) { std::cerr << std::endl << "ERROR: Parameter '" << name << "' not found in clf::Parameters object." << std::endl << std::endl; }
    assert(it!=map.end());
    if( it->second.type()==typeid(TYPE) ) { return std::any_cast<TYPE>(it->second); }
    
    assert(false);
    return TYPE();
  }

  /// Get a parameter with a default value 
  /**
     @param[in] name The name of the parameters 
     @param[in] value The default value (if the parameter is not found in the map, return this value)
     \return The parameter's value
   */
  template<typename TYPE>
  inline TYPE Get(std::string const& name, TYPE const& value) const { 
    auto it = map.find(name);
    if( it==map.end() ) { return value; }
    assert(it!=map.end());
    //it->second.get();
    if( it->second.type()==typeid(TYPE) ) { return std::any_cast<TYPE>(it->second); }
    
    assert(false);
    return TYPE();
  }

private:

  /// A map from the parameter name to its value (can be any type)
  std::unordered_map<std::string, std::any> map;
};

} // namespace clf

#endif
