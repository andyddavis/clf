#ifndef SUPPORTPOINTCLOUDEXCEPTIONS_HPP_
#define SUPPORTPOINTCLOUDEXCEPTIONS_HPP_

#include "clf/CLFException.hpp"

namespace clf {
namespace exceptions {

/// The graph connected the points in a support point cloud is not connected
class SupportPointCloudNotConnected : virtual public CLFException {
public:

  SupportPointCloudNotConnected();

  virtual ~SupportPointCloudNotConnected() = default;

private:
};

/// There are not enough support points in the cloud
/**
One of the support point requires more nearest neighbors than are available.
*/
class SupportPointCloudNotEnoughPointsException : virtual public CLFException {
public:
  /**
  @param[in] numPoints The number of support points in the cloud
  @param[in] required The required number of nearest neighbors
  */
  SupportPointCloudNotEnoughPointsException(std::size_t const numPoints, std::size_t const required);

  virtual ~SupportPointCloudNotEnoughPointsException() = default;

  /// The number of support points in the cloud
  const std::size_t numPoints;

  /// The required number of nearest neighbors
  const std::size_t required;

private:
};

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
