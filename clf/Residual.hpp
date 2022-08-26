#ifndef RESIDUAL_HPP_
#define RESIDUAL_HPP_

#include "clf/PointCloud.hpp"

#include "clf/SystemOfEquations.hpp"

#include "clf/DensePenaltyFunction.hpp"

namespace clf {
/// Compute the residual \f$\mathcal{L}(u(x_i), x_i) - f(x_i)\f$ for points \f$\{ x_i \}_{i=1}^{m}\f$
class Residual : public DensePenaltyFunction {
public:

  /**
     @param[in] cloud The point cloud \f$\{ x_i \}_{i=1}^{m}\f$
     @param[in] func The local function defined in this domain
     @param[in] system The system of equations that we want to locally satisfy
     @param[in] para Parameters for the residual computation
   */
  Residual(std::shared_ptr<PointCloud> const& cloud, std::shared_ptr<LocalFunction> const& func, std::shared_ptr<SystemOfEquations> const& system, std::shared_ptr<const Parameters> const& para);

  /**
     @param[in] func The local function defined in this domain
     @param[in] system The system of equations that we want to locally satisfy
     @param[in] para Parameters for the residual computation
   */
  Residual(std::shared_ptr<LocalFunction> const& func, std::shared_ptr<SystemOfEquations> const& system, std::shared_ptr<const Parameters> const& para);

  virtual ~Residual() = default;

  /// Get the ID of the system (see clf::SystemOfEquations::id)
  /**
     \return The ID of the system (see clf::SystemOfEquations::id)
   */
  std::size_t SystemID() const;

  /// The number of points \f$m\f$ 
  /**
     \return The number of points \f$m\f$ 
   */
  std::size_t NumPoints() const;

  /// Get the \f$i^{\text{th}}\f$ local point 
  /**
     @param[in] ind The index of the point we want 
     \return The \f$i^{\text{th}}\f$ local point 
   */
  std::shared_ptr<Point> GetPoint(std::size_t const ind) const;

  /// The number of Monte Carlo points on the boundary
  /**
     \return The number of Monte Carlo points on the boundary
   */
  std::size_t NumBoundaryPoints() const;

  /// Get the \f$i^{\text{th}}\f$ boundary point
  /**
     @param[in] ind The index of the boundary point
     \return The \f$i^{\text{th}}\f$ boundary point
   */
  std::shared_ptr<Point> GetBoundaryPoint(std::size_t const ind) const;

  /// Evaluate the residual
  /**
     @param[in] beta The coefficients for the local function
     \return The residual evaluated at each local point
   */
  virtual Eigen::VectorXd Evaluate(Eigen::VectorXd const& beta) override;

    /// Evaluate the Jacobian
  /**
     @param[in] beta The coefficients for the local function
     \return The Jacobian evaluated at each local point
   */
  virtual Eigen::MatrixXd Jacobian(Eigen::VectorXd const& beta) override;

  /// Evaluate the weighted sum of the Hessians
  /**
     @param[in] beta The coefficients for the local function
     @param[in] weights The weights for the weighted sum
     \return The weighted sum of the Hessian evaluated at each local point
   */
  virtual Eigen::MatrixXd Hessian(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) override;
  
protected:

  /// The points \f$\{ x_i \in \mathcal{B}_{\delta}(x) \}_{i=1}^{m}\f$
  std::shared_ptr<PointCloud> cloud;

  /// The system of equations we want to locally satisfy
  std::shared_ptr<const SystemOfEquations> system;

  /// The local function \f$u\f$
  std::shared_ptr<LocalFunction> function;

private:
  
};
  
} // namespace clf

#endif
