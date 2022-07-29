#ifndef PARAMETERS_HPP_
#define PARAMETERS_HPP_

#include <unordered_map>
#include <string>
#include <variant>
#include <optional>
#include <assert.h>
#include <iostream>

namespace clf {
  
 /// A class that holds algorithm parameters
class Parameters {
public:

  /// A parameter can only be one of these types
  typedef std::variant<float, double, int, unsigned int, std::size_t, std::string> Parameter;

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
  void Add(std::string const& name, Parameter const& in);

  /// Get a parameter
  /**
     @param[in] name The name of the parameters 
     \return The parameter's value
   */
  template<typename TYPE>
  inline TYPE Get(std::string const& name) const { return Extract<TYPE>(Get<Parameter>(name)); }

  /// Get a parameter with a default value 
  /**
     @param[in] name The name of the parameters 
     @param[in] value The default value (if the parameter is not found in the map, return this value)
     \return The parameter's value
   */
  template<typename TYPE>
  inline TYPE Get(std::string const& name, TYPE const& value) const { return Extract<TYPE>(Get<Parameter>(name, value)); }

private:

  /// Convert the varient into the type we care about 
  /**
     @param[in] para The varient type 
     \return Return the value if it matches the type, otherwise return an invalid std::optional
   */
  template<typename TYPE>
  inline std::optional<TYPE> ExtractType(Parameter const& para) const {
    if( const TYPE* val = std::get_if<TYPE>(&para) ) { return *val; }

    return {};
  }

  /// Get the parameter as the correct type 
  /**
     @param[in] para The parameter variant 
     \return The parameter as the correct type
   */
  template<typename TYPE> 
  inline TYPE Extract(Parameter const& para) const {
    // maybe it is another scalar type and we can convert?
    if( const std::optional<float> val = ExtractType<float>(para) ) { return (TYPE)(*val); }
    if( const std::optional<double> val = ExtractType<double>(para) ) { return (TYPE)(*val); }
    if( const std::optional<int> val = ExtractType<int>(para) ) { return (TYPE)(*val); }
    if( const std::optional<unsigned int> val = ExtractType<unsigned int>(para) ) { return (TYPE)(*val); }
    if( const std::optional<std::size_t> val = ExtractType<std::size_t>(para) ) { return (TYPE)(*val); }

    // something went wrong
    assert(false);
    return TYPE();
  }

  /// A map from the parameter name to its value (can be any type)
  std::unordered_map<std::string, Parameter> map;
};

} // namespace clf

/// Get a parameter
/**
   @param[in] name The name of the parameters 
   \return The parameter's value
*/
template<>
inline clf::Parameters::Parameter clf::Parameters::Get<clf::Parameters::Parameter>(std::string const& name) const { 
  auto it = map.find(name);
  if( it==map.end() ) { std::cerr << std::endl << "ERROR: Parameter '" << name << "' not found in clf::Parameters object." << std::endl << std::endl; }
  assert(it!=map.end());
  return it->second;
}

/// Get a parameter with a default value 
/**
   @param[in] name The name of the parameters 
   @param[in] value The default value (if the parameter is not found in the map, return this value)
   \return The parameter's value
*/
template<>
inline clf::Parameters::Parameter clf::Parameters::Get<clf::Parameters::Parameter>(std::string const& name, clf::Parameters::Parameter const& value) const { 
  auto it = map.find(name);
  if( it==map.end() ) { return value; }
  assert(it!=map.end());
  return it->second;
}

/// Get the parameter as the correct type (specialized for std::string)
/**
   @param[in] para The parameter variant 
   \return The parameter as the correct type
*/
template<>
inline std::string clf::Parameters::Extract<std::string>(Parameter const& para) const {
  if( const std::optional<std::string> val = ExtractType<std::string>(para) ) { return *val; }

  assert(false);
  return std::string();
}

#endif
