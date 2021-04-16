#ifndef SUPPORTPOINTCLOUDEXCEPTIONS_HPP_
#define SUPPORTPOINTCLOUDEXCEPTIONS_HPP_

#include "clf/CLFException.hpp"

namespace clf {
namespace exceptions {

/// Make sure the support points have the correct input/output dimension
class SupportPointCloudDimensionException : virtual public CLFException {
public:

  /// Is the input or output dimension mismatching?
  enum Type {
    /// The input dimensions do not match
    INPUT,

    /// The output dimensions do not match
    OUTPUT
  };

  /**
  @param[in] type Is the dimension mismatch in the input or output dimension?
  @param[in] ind1 The index of the first support point
  @param[in] ind2 The index of the second support point
  */
  SupportPointCloudDimensionException(Type const& type, std::size_t const ind1, std::size_t const ind2);

  virtual ~SupportPointCloudDimensionException() = default;

  /// Is the input or output dimenstion mismatched?
  const Type type;

  /// The index of the first point with mismatched dimension
  std::size_t const ind1;

  /// The index of the second point with mismatched dimension
  std::size_t const ind2;

private:
};

} // namespace exceptions
} // namespace clf

#endif
