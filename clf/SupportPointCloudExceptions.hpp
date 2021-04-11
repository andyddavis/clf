#ifndef SUPPORTPOINTCLOUDEXCEPTIONS_HPP_
#define SUPPORTPOINTCLOUDEXCEPTIONS_HPP_

#include <exception>
#include <string>

namespace clf {

/// Make sure the support points have the correct input/output dimension
class SupportPointCloudDimensionException : virtual public std::exception {
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

  /// The error message that gets printed when this exception is thrown
  virtual const char* what() const noexcept;

  /// Is the input or output dimenstion mismatched?
  const Type type;

  /// The index of the first point with mismatched dimension
  std::size_t const ind1;

  /// The index of the second point with mismatched dimension
  std::size_t const ind2;

private:

  /// The printed error message when this exception is thrown
  std::string message = "UNKNOWN ERROR";
};

} // namespace clf

#endif
