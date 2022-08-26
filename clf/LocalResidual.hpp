#ifndef LOCALRESIDUAL_HPP_
#define LOCALRESIDUAL_HPP_

#include "clf/Residual.hpp"

namespace clf {

/// Compute the residual \f$\mathcal{L}(u(x_i), x_i) - f(x_i)\f$ for points \f$\{ x_i \in \mathcal{B}_{\delta}(x) \}_{i=1}^{m}\f$, where \f$\mathcal{B}_{\delta}(x)\f$ is a radius \f$\delta\f$ centered at \f$x\f$.
/**
   <B>Configuration Parameters:</B>
   See clf::Residual for additional configuration parameters
   <B>Configuration Parameters:</B>
   Parameter Key | Type | Default Value | Description |
   ------------- | ------------- | ------------- | ------------- |
   "NumLocalPoints"   | <tt>std::size_t</tt> | <tt>---</tt> | The number of local points \f$m\f$. This is a required parameter. |
*/
class LocalResidual : public Residual {
public:

  /**
     @param[in] func The local function defined in this domain
     @param[in] system The system of equations that we want to locally satisfy
     @param[in] para Parameters for the residual computation
   */
LocalResidual(std::shared_ptr<LocalFunction> const& func, std::shared_ptr<SystemOfEquations> const& system, std::shared_ptr<const Parameters> const& para);

  virtual ~LocalResidual() = default;

private:

  /// Generate the local points
  /**
     @param[in] func The local function 
     @param[in] num The number of points \f$T\f$
     \return The points \f$\{ x_i \in \mathcal{B}_{\delta}(x) \}_{i=1}^{m}\f$
   */
  static std::shared_ptr<PointCloud> GeneratePoints(std::shared_ptr<LocalFunction> const& func, std::size_t const num);

};

} // namespace clf

#endif
