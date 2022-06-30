#ifndef BASISFUNCTIONS_HPP_
#define BASISFUNCTIONS_HPP_

#include <cstddef>

#include <Eigen/Eigen>

namespace clf {

/// A basis function \f$\varphi_p: \mathcal{D} \mapsto \mathbb{R}\f$ such that \f$\langle \varphi_p, \varphi_q \rangle_{\mathcal{D}} = 1\f$ if \f$p=q\f$ and \f$0\f$ otherwise. 
/**
This class implements an orgothonal basis function \f$\varphi_p: \mathcal{D} \mapsto \mathbb{R}\f$ such that
\f{equation*}{
   \langle \varphi_p, \varphi_q \rangle_{\mathcal{D}} = \int_{\mathcal{D}} \varphi_p \varphi_q \, dx = \begin{cases}
   1 & \text{if } p=q \\
   0 & \text{else,}
   \end{cases}
\f}
where \f$\mathcal{D} \subseteq \mathbb{R}\f$. 
*/
class BasisFunctions {
public:

  BasisFunctions();

  virtual ~BasisFunctions() = default;
  
  /// Evaluate the basis function \f$\varphi_{p}(x)\f$.
  /**
     @param[in] p The order of the basis function 
     @param[in] x Evaluate the \f$p^{\text{th}}\f$ basis function at this point
     \return The basis function \f$\varphi_{p}(x)\f$.
  */
  virtual double Evaluate(int const p, double const x) const = 0;

  /// Evaluate the basis functions \f$[\varphi_{0}(x),\, ...,\, \varphi_{p}(x)]\f$.
  /**
     @param[in] p Evaluate up to the \f$p^{\text{th}}\f$ order of the basis function 
     @param[in] x Evaluate the basis functions at this point
     \return The vector of basis functions \f$[\varphi_{0}(x),\, ...,\, \varphi_{p}(x)]\f$.
  */
  virtual Eigen::VectorXd EvaluateAll(int const p, double const x) const;

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
     @param[in] p Evaluate up to the \f$p^{\text{th}}\f$ order of the basis function 
     @param[in] x Evaluate the basis functions at this point
     @param[in] k Evaluate up to the \f$k^{\text{th}}\f$ derivative
     \return The matrix \f$D\f$ of basis function derivatives.
  */
  virtual Eigen::MatrixXd EvaluateAllDerivatives(int const p, double const x, std::size_t const k) const;

  /// Evaluate the basis function derivative \f$\frac{d^{k}}{dx^{k}} \varphi_{p}(x)\f$.
  /**
     @param[in] p The order of the basis function 
     @param[in] x Evaluate the \f$p^{\text{th}}\f$ basis function derivative at this point
     @param[in] k Evaluate the \f$k^{\text{th}}\f$ derivative
     \return The basis function derivative \f$\frac{d^{k}}{dx^{k}} \varphi_{p}(x)\f$.
  */
  virtual double EvaluateDerivative(int const p, double const x, std::size_t const k) const = 0;

private:
};

} // namespace clf 

#endif
