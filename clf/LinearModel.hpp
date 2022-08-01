#ifndef LINEARMODEL_HPP_
#define LINEARMODEL_HPP_

#include <optional>

#include "clf/SystemOfEquations.hpp"

namespace clf {

/// A system of equations with the form \f$A(x) u(x) - f(x) = 0\f$, where \f$u(x) = \Phi(x)^{\top} c\f$ and \f$A \in \mathbb{R}^{m \times \hat{m}}\f$.
/**
   This model defaults to \f$A(x)\f$ being the identity, but if that is the desired model then clf::IdentityModel might be a slightly more efficient choice.
*/
class LinearModel : public SystemOfEquations {
public:

  /**
     Set \f$\hat{m} = m\f$.
     @param[in] indim The input dimension \f$d\f$
     @param[in] outdim The output dimension \f$m\f$
     @param[in] para The parameters for this system of equations
   */
  LinearModel(std::size_t const indim, std::size_t const outdim, std::shared_ptr<const Parameters> const& para = std::make_shared<Parameters>());

  /**
   <B>Configuration Parameters:</B>
   Parameter Key | Type | Default Value | Description |
   ------------- | ------------- | ------------- | ------------- |
   "InputDimension"   | <tt>std::size_t</tt> | --- | The input dimension \f$d\f$ of the local function. This is a required parameter. |
   "OutputDimension"   | <tt>std::size_t</tt> | --- | The output dimension \f$m\f$ of the local function. This is a required parameter. |
   "LocalFunctionOutputDimension"   | <tt>std::size_t</tt> | <tt>output dimension</tt> |  The number of columns \f$\hat{m}\f$ in \f$A(x)\f$. Defaults to the output dimension. |
     @param[in] para The parameters for this system of equations
   */
  LinearModel(std::shared_ptr<const Parameters> const& para);

  /**
     @param[in] indim The input dimension \f$d\f$
     @param[in] outdim The output dimension \f$m\f$
     @param[in] matdim The number of columns \f$\hat{m}\f$ in \f$A(x)\f$
     @param[in] para The parameters for this system of equations
   */
  LinearModel(std::size_t const indim, std::size_t const outdim, std::size_t const matdim, std::shared_ptr<const Parameters> const& para = std::make_shared<const Parameters>());

  /**
   <B>Configuration Parameters:</B>
   Parameter Key | Type | Default Value | Description |
   ------------- | ------------- | ------------- | ------------- |
   "InputDimension"   | <tt>std::size_t</tt> | --- | The input dimension \f$d\f$ of the local function. This is a required parameter. |
     @param[in] A The matrix \f$A \in \mathbb{R}^{m \times \hat{m}}\f$
     @param[in] para The parameters for this system of equations
   */
  LinearModel(Eigen::MatrixXd const& A, std::shared_ptr<const Parameters> const& para);

  /**
     @param[in] indim The input dimension \f$d\f$
     @param[in] A The matrix \f$A \in \mathbb{R}^{m \times \hat{m}}\f$
     @param[in] para The parameters for this system of equations
   */
  LinearModel(std::size_t const indim, Eigen::MatrixXd const& A, std::shared_ptr<const Parameters> const& para = std::make_shared<const Parameters>());

  virtual ~LinearModel() = default;

  /// Evaluate the matrix \f$A(x) \in \mathbb{R}^{m \times \hat{m}}\f$.
  /**
     Default to the \f$m \times \hat{m}\f$ identity matrix. 
     @param[in] x The location \f$x\f$
     \return The matrix \f$A(x) \in \mathbb{R}^{m \times \hat{m}}\f$.
   */
  virtual Eigen::MatrixXd Operator(Eigen::VectorXd const& x) const;

  /// Evaluate the operator given the function \f$u\f$ and at the location \f$x\f$.
  /**
     Implements \f$\mathcal{L}(u(x), x) = A(x) u(x)\f$.
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     \return The operator evaluation  \f$\mathcal{L}(u(x), x)\f$
   */
  virtual Eigen::VectorXd Operator(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const final override; 

  /// Evaluate the Jacobian of the operator with respect to the cofficients \f$c\f$, \f$\nabla_{c} \mathcal{L}(u(x; c), x)\f$ given the function \f$u\f$ and at the location \f$x\f$.
  /**
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     \return The operator evaluation  \f$\mathcal{L}(u(x), x)\f$
   */
  virtual Eigen::MatrixXd JacobianWRTCoefficients(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const final override;

  /// Compute the weighted sum of the Hessian of each output of the operator with respect to the cofficients \f$c\f$, \f$\sum_{i=1}^{m} w_i \nabla_{c}^2 \mathcal{L}_i(u(x; c), x)\f$ given the function \f$u\f$ and at the location \f$x\f$.
  /**
     @param[in] u The function \f$u\f$
     @param[in] x The location \f$x\f$
     @param[in] coeff The coefficients \f$c\f$ that define the location function 
     @param[in] wieghts The weights for the sum
     \return The operator Hessian \f$\sum_{i=1}^{m} w_i \nabla_{c}^2 \mathcal{L}_i(u(x; c), x)\f$
   */
  virtual Eigen::MatrixXd HessianWRTCoefficients(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff, Eigen::VectorXd const& weights) const final override;

private:

  /// The number of columns \f$\hat{m}\f$ in \f$A(x)\f$
  const std::size_t matdim;

  /// The matrix \f$A \in \mathbb{R}^{m \times \hat{m}}\f$
  /**
     Stored optionally if it does not depend on \f$x\f$
   */
  const std::optional<const Eigen::MatrixXd> A;
};

} // namespace clf

#endif
