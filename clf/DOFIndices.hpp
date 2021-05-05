#ifndef DOFINDICES_HPP_
#define DOFINDICES_HPP_

#include "clf/SupportPointCloud.hpp"

namespace clf {

/// A map from the global support point number and its local degree of freedom indices to indices for the global degrees of freedom
class DOFIndices {
public:

  /**
  @param[in] cloud The support point cloud that holds all of the points (and their local cost functions)
  */
  DOFIndices(std::shared_ptr<SupportPointCloud> const& cloud);

  virtual ~DOFIndices() = default;

  /// The support point cloud that holds all of the points (and their local cost functions)
  std::shared_ptr<const SupportPointCloud> cloud;

  /// Each entry corresponds to a support point, it is the index of the first global DOF
  const std::vector<std::size_t> globalDoFIndices;

  /// The maximum number of nonzero terms in the Hessian matrix of the global cost
  const std::size_t maxNonZeros;

private:

  /// Compute the global degrees of freedom for each support point
  /**
  @param[in] cloud The support point cloud that holds all of the points (and their local cost functions)
  \return The global degrees of freedom for each support point
  */
  static std::vector<std::size_t> GlobalDOFIndices(std::shared_ptr<SupportPointCloud> const& cloud);

  /// Compute the maximum number of nonzero terms in the Hessian matrix of the global cost
  /**
  @param[in] cloud The support point cloud that holds all of the points (and their local cost functions)
  \return The maximum number of nonzero terms in the Hessian matrix of the global cost
  */
  static std::size_t MaxNonZeros(std::shared_ptr<SupportPointCloud> const& cloud);
};

} // namespace clf

#endif
