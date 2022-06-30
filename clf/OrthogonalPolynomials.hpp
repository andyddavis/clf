#ifndef ORTHOGONALPOLYNOMIALS_HPP_
#define ORTHOGONALPOLYNOMIALS_HPP_

#include "clf/BasisFunctions.hpp"

namespace clf {

/// Evaluate orthogonal polynomial basis functions \f$\varphi_p: \mathcal{D} \mapsto \mathbb{R}\f$ of order \f$p\f$.
/**
We use the recursive relationship to evaluate the \f$p^{\text{th}}\f$ degree polynomial
\f{eqnarray*}{
   \varphi_0(x) &=& \varphi_0 \\
   \varphi_1(x) &=& \varphi_1(x) \\
   \varphi_p(x) &=& (A_p x + B_p) \varphi_{p-1}(x) - C_p \varphi_{p-2}(x) \quad \text{for} \quad p>1.
\f}
The \f$k^{\text{th}}\f$ derivative is
\f{eqnarray*}{
   \frac{d^{k}}{d x^{k}}\varphi_0(x) &=& 0 \quad \text{for all} \quad k>0 \\
   \frac{d^{k}}{d x^{k}}\varphi_1(x) &=& \varphi_1 \quad \text{for} \quad k=1 \quad \text{and} \quad \frac{d^{k}}{d x^{k}}\varphi_1(x) = 0 \quad \text{for all} \quad k>1 \\
   \frac{d^{k}}{d x^{k}}\varphi_p(x) &=& 0 \quad \text{for} \quad p>1,\, k > p \quad \text{and} \quad \frac{d^{k}}{d x^{k}}\varphi_p(x) = k A_p \frac{d^{k-1}}{d x^{k-1}}\varphi_{p-1}(x) + (A_p x + B_p) \frac{d^{k}}{d x^{k}}\varphi_{p-1}(x) - C_p \frac{d^{k}}{d x^{k}}\varphi_{p-2}(x) \quad \text{for} \quad p>1,\, 0 < k \leq p,
\f}
using the notation that \f$\frac{d^0}{dx^0} \varphi_p(x) = \varphi_p(x)\f$.
*/
class OrthogonalPolynomials : public BasisFunctions {
public:

  OrthogonalPolynomials();

  virtual ~OrthogonalPolynomials() = default;

  /// Evaluate the basis function \f$\varphi_{p}(x)\f$.
  /**
     @param[in] p The order of the basis function 
     @param[in] x Evaluate the \f$p^{\text{th}}\f$ basis function at this point
     \return The basis function \f$\varphi_{p}(x)\f$.
  */
  virtual double Evaluate(int const p, double const x) const final override;

  /// Evaluate the basis functions \f$[\varphi_{0}(x),\, ...,\, \varphi_{p}(x)]\f$.
  /**
     Override this function so we do not unnecessarily keep calling the recursive evaluate function.
     @param[in] p Evaluate up to the \f$p^{\text{th}}\f$ order of the basis function 
     @param[in] x Evaluate the basis functions at this point
     \return The vector of basis functions \f$[\varphi_{0}(x),\, ...,\, \varphi_{p}(x)]\f$.
  */
  virtual Eigen::VectorXd EvaluateAll(int const p, double const x) const override;

  /// Evaluate the basis function derivative \f$\frac{d^{k}}{dx^{k}} \varphi_{p}(x)\f$.
  /**
     @param[in] p The order of the basis function 
     @param[in] x Evaluate the \f$p^{\text{th}}\f$ basis function derivative at this point
     @param[in] k Evaluate the \f$k^{\text{th}}\f$ derivative
     \return The basis function derivative \f$\frac{d^{k}}{dx^{k}} \varphi_{p}(x)\f$.
  */
  virtual double EvaluateDerivative(int const p, double const x, std::size_t const k) const final override;

  /// Evaluate the up to the \f$k^{\text{th}}\f$ derivative of the \f$0,\, ....,\, p\f$ basis functions
  /**
     Compute the matrix 
     \f{equation*}{
     D = \begin{bmatrix}
     \frac{d}{dx} \varphi_0(x) & ... & \frac{d^k}{dx^k} \varphi_0(x) \\
     \vdots & \ddots & \vdots \\
     \frac{d}{dx} \varphi_p(x) & ... & \frac{d^k}{dx^k} \varphi_p(x) \\
     \end{bmatrix}.
     \f}
     Override this function so we do not unnecessarily keep calling the recursive evaluate function.
     @param[in] p Evaluate up to the \f$p^{\text{th}}\f$ order of the basis function 
     @param[in] x Evaluate the basis functions at this point
     @param[in] k Evaluate up to the \f$k^{\text{th}}\f$ derivative
     \return The matrix \f$D\f$ of basis function derivatives.
  */
  virtual Eigen::MatrixXd EvaluateAllDerivatives(int const p, double const x, std::size_t const k) const override;

private:

  /// Evaluate the constant orthogonal polynomial
  /**
     \return The first orthogonal polynomial \f$\varphi_0\f$, which is constant with respect to \f$x\f$.
   */
  virtual double Phi0() const = 0;

  /// Evaluate the linear orthogonal polynomial
  /**
     @param[in] x Evaluate the linear basis function derivative at this point
     \return The second orthogonal polynomial \f$\varphi_1\f$, which is linear with respect to \f$x\f$.
   */
  virtual double Phi1(double const x) const = 0;

  /// Evaluate the derivative of the linear orthogonal polynomial
  /**
     \return The derivative of the second orthogonal polynomial \f$\frac{d}{dx} \phi_1\f$, which is constant with respect to \f$x\f$.
   */
  virtual double dPhi1dx() const = 0;

  /// The constant \f$A_p\f$ in the recurrance relationship.
  /**
     @param[in] p The order \f$p\f$.
     \return The constant \f$A_p\f$.
   */
  virtual double Ap(std::size_t const p) const = 0;

  /// The constant \f$B_p\f$ in the recurrance relationship.
  /**
     @param[in] p The order \f$p\f$.
     \return The constant \f$B_p\f$.
   */
  virtual double Bp(std::size_t const p) const = 0;

  /// The constant \f$C_p\f$ in the recurrance relationship.
  /**
     @param[in] p The order \f$p\f$.
     \return The constant \f$C_p\f$.
   */
  virtual double Cp(std::size_t const p) const = 0;

};

};

#endif 
