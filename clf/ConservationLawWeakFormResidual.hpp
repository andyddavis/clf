#ifndef CONSERVATIONLAWWEAKFORMRESIDUAL_HPP_
#define CONSERVATIONLAWWEAKFORMRESIDUAL_HPP_

#include "clf/PenaltyFunction.hpp"
#include "clf/PointCloud.hpp"
#include "clf/ConservationLaw.hpp"

namespace clf {

/// The weak form residual of a clf::ConservationLaw
/**
The strong form of the conservation law is \f$\nabla_x \cdot F(u(\cdot), x) - f(x) = 0\f$. For <em>any</em> test function \f$\varphi\f$ and compact domain \f$\mathcal{B}\f$, the corresponding weak form is 
\f{equation*}{
\int_{\partial \mathcal{B}} \varphi F \cdot n \, dx - \int_{\mathcal{B}} \nabla_{x} \varphi \cdot F + \varphi f \, dx = 0.
\f}
If we represent \f$\varphi\f$ as a clf::LocalFunction with output dimension \f$m=1\f$. Therefore, \f$\varphi = \phi^{\top} a\f$, where \f$\phi \in \mathbb{R}^{n}\f$ is a vector of basis functions. The weak form becomes
\f{equation*}{
\sum_{i=1}^{n} a_i \left( \int_{\partial \mathcal{B}} \phi_i F \cdot n \, dx - \int_{\mathcal{B}} \nabla_{x} \phi_i \cdot F + \phi_i f \, dx \right) = 0.
\f}
Since this must be true for <em>any</em> test function, this must be true for <em>any</em> vector of coefficients \f$a\f$. We, therefore, define the \f$n\f$ penalty functions 
\f{equation*}{
c_i(\beta) = \int_{\partial \mathcal{B}} \phi_i F(u(\cdot; \beta), x) \cdot n \, dx - \int_{\mathcal{B}} \nabla_{x} \phi_i \cdot F(u(\cdot; \beta), x) + \phi_i f \, dx 
\f}

<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"NumPoints"   | <tt>std::size_t</tt> | <tt>---</tt> | The number of Monte Carlo points \f$T\f$. This is a required parameter. |
*/
class ConservationLawWeakFormResidual : public DensePenaltyFunction {
public:
  /**
     @param[in] func The function \f$u(\cdot, \beta\f$
     @param[in] system The conservation law we are trying to satisfy
     @param[in] testFunctionBasis The feature vector that defines the test function 
     @param[in] para The parameters for this penalty function 
  */
  ConservationLawWeakFormResidual(std::shared_ptr<LocalFunction> const& func, std::shared_ptr<ConservationLaw> const& system, std::shared_ptr<FeatureVector> const& testFunctionBasis, std::shared_ptr<const Parameters> const& para);

  virtual ~ConservationLawWeakFormResidual() = default;

  /// Evaluate the penalty function \f$c: \mathbb{R}^{d} \mapsto \mathbb{R}^{n}\f$
  /**
     There are \f$n\f$ penalty functions 
     \f{equation*}{
     c_i(\beta) = \int_{\partial \mathcal{B}} \phi_i F(u(\cdot; \beta), x) \cdot n \, dx - \int_{\mathcal{B}} \nabla_{x} \phi_i \cdot F(u(\cdot; \beta), x) + \phi_i f \, dx,
     \f}
     which we approximate with Monte Carlo 
     \f{equation*}{
     c_i(\beta) = \frac{1}{T} \sum_{j=1}^{T} \phi_i(x_j) F(u(\cdot; \beta), x_j) \cdot n - \frac{1}{T} \sum_{j=1}^{T} \left( \nabla_{x} \phi_i(x_j) \cdot F(u(\cdot; \beta), x_j) + \phi_i(x_j) f(x_j) \right)
     \f}
     where \f$x_j \sim U(\partial \mathcal{B})\f$ and \f$x_j \sim U(\mathcal{B})\f$ are sampled from uniform distributions over the boundary \f$\partial \mathcal{B}\f$ and the domain \f$\mathcal{B}\f$.
     @param[in] beta The input parameters \f$\beta\f$
     \return The penalty function evaluation \f$c(\beta)\f$
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

  /// The number of Monte Carlo points on the boundary
  /**
     \return The number of Monte Carlo points on the boundary
   */
  std::size_t NumBoundaryPoints() const;

  /// Get the \f$i^{\text{th}}\f$ Monte Carlo point
  /**
     @param[in] ind The index of the point
     \return The \f$i^{\text{th}}\f$ Monte Carlo point
   */
  std::shared_ptr<Point> GetPoint(std::size_t const ind) const;

  /// The number of Monte Carlo points in the interior of the domain
  /**
     \return The number of Monte Carlo points in the interior of the domain
   */
  std::size_t NumPoints() const;

  /// Get the \f$i^{\text{th}}\f$ boundary point
  /**
     @param[in] ind The index of the boundary point
     \return The \f$i^{\text{th}}\f$ boundary point
   */
  std::shared_ptr<Point> GetBoundaryPoint(std::size_t const ind) const;

private:
  
  /// Generate the Monte Carlo points
  /**
     @param[in] num The number of points \f$T\f$
   */
  void GeneratePoints(std::size_t const num);
  
  /// The local function \f$u\f$
  std::shared_ptr<LocalFunction> function;

  /// The conservation law we are trying to satisfy
  std::shared_ptr<ConservationLaw> system;

  // The feature vector that defines the test function
  std::shared_ptr<FeatureVector> testFunctionBasis;

  /// The Monte Carlo
  std::shared_ptr<PointCloud> points;
};
  
} // namespace clf

#endif
