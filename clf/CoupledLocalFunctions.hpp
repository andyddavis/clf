#ifndef COUPLEDLOCALFUNCTION_HPP_
#define COUPLEDLOCALFUNCTION_HPP_

#include "clf/PointCloud.hpp"

#include "clf/LocalFunction.hpp"

namespace clf {

/// A function \f$u: \Omega \mapsto \mathbb{R}^{m}\f$, where \f$\Omega \subseteq \mathbb{R}^{d}\f$
class CoupledLocalFunctions {
public:

  /**
     @param[in] cloud The point cloud
  */
  CoupledLocalFunctions(std::shared_ptr<PointCloud> const& cloud);

  virtual ~CoupledLocalFunctions() = default;

  /// The number of local functions
  std::size_t NumLocalFunctions() const;
  
private:

  /// The point cloud
  /**
     Each point is associated with a clf::LocalFunction
  */
  std::shared_ptr<PointCloud> cloud;
  
};
  
} // namespace clf

#endif
