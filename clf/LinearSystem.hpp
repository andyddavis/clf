#ifndef LINEARSYSTEM_HPP_
#define LINEARSYSTEM_HPP_

#include <optional>

#include "clf/SystemOfEquations.hpp"

namespace clf {

/// A system of equations with the form \f$A(x) u(x) - f(x) = 0\f$, where \f$u(x) = \Phi(x)^{\top} c\f$ and \f$A \in \mathbb{R}^{m \times \hat{m}}\f$.
/**
   Let \f$u: \Omega \mapsto \mathbb{R}^{\hat{m}}\f$, where \f$\Omega \subseteq \mathbb{R}^{d}\f$. The system of equations is composed of two parts. 
*/
class LinearSystem : public SystemOfEquations {
public:

  /**
     Set \f$\hat{m} = m\f$.
     @param[in] indim The input dimension \f$d\f$
     @param[in] outdim The output dimension \f$m\f$
   */
  LinearSystem(std::size_t const indim, std::size_t const outdim);

  /**
     @param[in] indim The input dimension \f$d\f$
     @param[in] outdim The output dimension \f$m\f$
     @param[in] matdim The number of columns \f$\hat{m}\f$ in \f$A(x)\f$
   */
  LinearSystem(std::size_t const indim, std::size_t const outdim, std::size_t const matdim);

  /**
     @param[in] indim The input dimension \f$d\f$
     @param[in] A The matrix \f$A \in \mathbb{R}^{m \times \hat{m}}\f$
   */
  LinearSystem(std::size_t const indim, Eigen::MatrixXd const A);

  virtual ~LinearSystem() = default;

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
