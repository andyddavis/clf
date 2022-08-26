#ifndef COUPLEDLOCALFUNCTION_HPP_
#define COUPLEDLOCALFUNCTION_HPP_

#include "clf/PointCloud.hpp"

#include "clf/LocalFunction.hpp"

#include "clf/BoundaryCondition.hpp"

namespace clf {

/// A function \f$u: \Omega \mapsto \mathbb{R}^{m}\f$, where \f$\Omega \subseteq \mathbb{R}^{d}\f$
/**
*/
class CoupledLocalFunctions {
public:

  /**
     <B>Additional Configuration Parameters:</B>
     Parameter Key | Type | Default Value | Description |
     ------------- | ------------- | ------------- | ------------- |
     "NumSupportPoints"   | <tt>std::size_t</tt> | --- | The number of support points. This is a required parameter |
     
     @param[in] domain The domain \f$\Omega\f$
     @param[in] delta The local domains are \f$x \in \mathcal{B}_i\f$ if \f$x_j \in [y_i-\delta_j, y_i+\delta_j]\f$ and \f$x \in \Omega\f$, where \f$y \in \Omega\f$ are the support points
     @param[in] para Parameters for this coupled local function
  */
  CoupledLocalFunctions(std::shared_ptr<MultiIndexSet> const& set, std::shared_ptr<BasisFunctions> const& basis, std::shared_ptr<Domain> const& domain, Eigen::VectorXd const& delta, std::shared_ptr<Parameters> const& para);

  virtual ~CoupledLocalFunctions() = default;

  /// The number of local functions
  std::size_t NumLocalFunctions() const;

  /// Set Dirichlet boundary conditions
  /**
     @param[in] system The system that defines the boundary condition
     @param[in] func A function that return <tt>true</tt> if the proposed point is on the boundary where this condition is to be enforced, otherwise it returns <tt>false</tt>. The input to the function is a pair, the first vector is the point on the boundary and the second vector is the outward pointing normal.
     @param[in] numPoints The number of points to use enforce this boundary condition
   */
  void SetBoundaryCondition(std::shared_ptr<SystemOfEquations> const& system, std::function<bool(std::pair<Eigen::VectorXd, Eigen::VectorXd> const&)> const& func, std::size_t const numPoints);

  typedef std::vector<std::shared_ptr<BoundaryCondition> > BoundaryConditions;

  std::optional<BoundaryConditions> GetBCs(std::size_t const ind) const;
  
private:

  /// The global domain
  std::shared_ptr<Domain> domain;

  /// The point cloud
  /**
     Each point is associated with a clf::LocalFunction
  */
  std::shared_ptr<PointCloud> cloud;

  /// A map from clf::Point::id to boundary conditions, not all points will have boundary conditions associated with them
  std::unordered_map<std::size_t, BoundaryConditions> boundaryConditions;

  /// A map from the clf::Point::id to the clf::LocalFunction associated with that support point
  std::unordered_map<std::size_t, std::shared_ptr<LocalFunction> > functions;
  
};
  
} // namespace clf

#endif
