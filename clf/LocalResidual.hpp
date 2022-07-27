#ifndef LOCALRESIDUAL_HPP_
#define LOCALRESIDUAL_HPP_

#include "clf/PointCloud.hpp"

#include "clf/SystemOfEquations.hpp"

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
     @param[in] system The system of equations that we want to locally satisfy
     @param[in] point The center point for the ball \f$\mathcal{B}_{\delta}(x)\f$
     @param[in] para Parameters for the residual computation
   */
LocalResidual(std::shared_ptr<LocalFunction> const& func, std::shared_ptr<SystemOfEquations> const& system, Point const& point, std::shared_ptr<const Parameters> const& para = std::make_shared<Parameters>());

  virtual ~LocalResidual() = default;

  /// Evaluate the residual
  /**
     @param[in] beta The coefficients for the local function
     \return The residual evaluated at each local point
   */
  virtual Eigen::VectorXd Evaluate(Eigen::VectorXd const& beta) final override;

  /// Evaluate the Jacobian
  /**
     @param[in] beta The coefficients for the local function
     \return The Jacobian evaluated at each local point
   */
  virtual Eigen::MatrixXd Jacobian(Eigen::VectorXd const& beta) final override;

  /// Evaluate the weighted sum of the Hessians
  /**
     @param[in] beta The coefficients for the local function
     @param[in] weights The weights for the weighted sum
     \return The weighted sum of the Hessian evaluated at each local point
   */
  virtual Eigen::MatrixXd Hessian(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) final override;

  /// The number of local points \f$m\f$ 
  /**
     \return The number of local points \f$m\f$ 
   */
  std::size_t NumLocalPoints() const; 

  /// Get the \$i^{\text{th}}\f$ local point 
  /**
     @param[in] ind The index of the point we want 
     \return The \$i^{\text{th}}\f$ local point 
   */
  Point GetPoint(std::size_t const ind) const;

private:

  /// Generate the local points
  /**
     @param[in] point The center point for the ball \f$\mathcal{B}_{\delta}(x)\f$
     @param[in] num The number points \f$m\f$
     @param[in] delta The radius of the local ball \f$\delta\f$
     \return The points \f$\{ x_i \in \mathcal{B}_{\delta}(x) \}_{i=1}^{m}\f$
   */
  static PointCloud GeneratePoints(Point const& point, std::size_t const num, double const delta);

  /// The points \f$\{ x_i \in \mathcal{B}_{\delta}(x) \}_{i=1}^{m}\f$
  const PointCloud points;

  /// The local function \f$u\f$
  std::shared_ptr<LocalFunction> function;

  /// The system of equations we want to locally satisfy
  std::shared_ptr<const SystemOfEquations> system;
};

} // namespace clf

#endif
