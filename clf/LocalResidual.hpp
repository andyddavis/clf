#ifndef LOCALRESIDUAL_HPP_
#define LOCALRESIDUAL_HPP_

#include "clf/PenaltyFunction.hpp"

namespace clf {

/// Compute the residual \f$\mathcal{L}(u(x_i), x_i) - f(x_i)\f$ for points \f$\{ x_i \in \mathcal{B}_{\delta}(x) \}_{i=1}^{m}\f$, where \f$\mathcal{B}_{\delta}(x)\f$ is a radius \f$\delta\f$ centered at \f$x\f$.
/**
   <B>Configuration Parameters:</B>
   Parameter Key | Type | Default Value | Description |
   ------------- | ------------- | ------------- | ------------- |
   "Radius"   | <tt>double</tt> | <tt>---</tt> | The radius of the local ball \f$\delta\f$. This is a required parameter. |
   "NumPoints"   | <tt>std::size_t</tt> | <tt>---</tt> | The number of local points \f$m\f$. This is a required parameter. |
*/
class LocalResidual : public DensePenaltyFunction {
public:

  /**
     @param[in] point The center point for the ball \f$\mathcal{B}_{\delta}(x)\f$
     @param[in] para Parameters for the residual computation
   */
  LocalResidual(Eigen::VectorXd const& point, std::shared_ptr<const Parameters> const& para = std::make_shared<Parameters>());

  virtual ~LocalResidual() = default;

  /// Evaluate the residual
  /**
     @param[in] beta The coefficients for the local function
     \return The residual evaluated at each local point
   */
  virtual Eigen::VectorXd Evaluate(Eigen::VectorXd const& beta) override;

  /// The number of local points \f$m\f$ 
  std::size_t NumLocalPoints() const; 

private:

  /// Generate the local points
  /**
     @param[in] point The center point for the ball \f$\mathcal{B}_{\delta}(x)\f$
     @param[in] num The number points \f$m\f$
     @param[in] delta The radius of the local ball \f$\delta\f$
     \return The points \f$\{ x_i \in \mathcal{B}_{\delta}(x) \}_{i=1}^{m}\f$
   */
  static std::vector<Eigen::VectorXd> GeneratePoints(Eigen::VectorXd const& point, std::size_t const num, double const delta);

  /// The points \f$\{ x_i \in \mathcal{B}_{\delta}(x) \}_{i=1}^{m}\f$
  const std::vector<Eigen::VectorXd> points;
};

} // namespace clf

#endif
