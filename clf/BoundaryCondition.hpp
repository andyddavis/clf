#ifndef BOUNDARYCONDITION_HPP_
#define BOUNDARYCONDITION_HPP_

#include "clf/Residual.hpp"

namespace clf {

/**
   <B>Configuration Parameters:</B>
   See clf::Residual for additional configuration parameters
*/
class BoundaryCondition : public Residual {
public:

   /**
     @param[in] func The local function defined in this domain
     @param[in] system The system of equations that we want to locally satisfy
     @param[in] para Parameters for the residual computation
   */
  BoundaryCondition(std::shared_ptr<LocalFunction> const& func, std::shared_ptr<SystemOfEquations> const& system, std::shared_ptr<const Parameters> const& para = std::make_shared<Parameters>());

  virtual ~BoundaryCondition() = default;

  /// Add a point to the point cloud where we enforce the system of equations
  /**
     Note that this is a point on the boundary of a domain, but not necessarily the domain where the local function is defined. For example, this may be a boundary point on the super set of the local function's domain. 

     @param[in] pnt The point on the boundary of a domain
   */
  void AddPoint(std::pair<Eigen::VectorXd, Eigen::VectorXd> const& pnt);
  
private:
};
} // namespace clf

#endif
