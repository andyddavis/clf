#ifndef MODEL_HPP_
#define MODEL_HPP_

#include <boost/property_tree/ptree.hpp>

namespace clf {

/// Implements a model that is given to a clf::SupportPoint
/**
<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"InputDimension"   | <tt>std::size_t</tt> | <tt>1</tt> | The input dimension. |
"OutputDimension"   | <tt>std::size_t</tt> | <tt>1</tt> | The output dimension. |
*/
class Model {
public:

  /**
  @param[in] pt Options for the model
  */
  Model(boost::property_tree::ptree const& pt);

  virtual ~Model() = default;

  /// The input dimension \f$d\f$
  const std::size_t inputDimension;

  /// The output dimension \f$m\f$
  const std::size_t outputDimension;

private:
};

} // namespace clf

#endif
