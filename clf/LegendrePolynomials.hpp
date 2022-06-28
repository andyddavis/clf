#ifndef LEGENDREPOLYNOMIALS_HPP_
#define LEGENDREPOLYNOMIALS_HPP_

#include "clf/OrthogonalPolynomials.hpp"

namespace clf {

/// Orthogonal polynomials \f$\phi_p: \mathcal{D} \mapsto \mathbb{R}\f$ over the domain \f$\mathcal{D} = [-1, 1]\f$.
class LegendrePolynomials : public OrthogonalPolynomials {
public:

  LegendrePolynomials();

  virtual ~LegendrePolynomials() = default;

private:

  /// Evaluate the constant orthogonal polynomial
  /**
     \return The first orthogonal polynomial is \f$\phi_0(x) = 1\f$.
   */
  virtual double Phi0() const final override;

  /// Evaluate the linear orthogonal polynomial
  /**
     @param[in] x Evaluate the linear basis function derivative at this point
     \return The second orthogonal polynomial is \f$\phi_1(x) = x\f$.
   */
  virtual double Phi1(double const x) const final override;

  /// Evaluate the derivative of the linear orthogonal polynomial
  /**
     \return The derivative of the second orthogonal polynomial \f$\frac{d}{dx} \phi_1 = 1\f$.
   */
  virtual double dPhi1dx() const final override;

  /// The constant \f$A_p\f$ in the recurrance relationship.
  /**
     @param[in] p The order \f$p\f$.
     \return The constant \f$A_p = (2p-1)/p\f$.
   */
  virtual double Ap(std::size_t const p) const final override;

  /// The constant \f$B_p\f$ in the recurrance relationship.
  /**
     @param[in] p The order \f$p\f$.
     \return The constant \f$B_p=0\f$.
   */
  virtual double Bp(std::size_t const p) const final override;

  /// The constant \f$C_p\f$ in the recurrance relationship.
  /**
     @param[in] p The order \f$p\f$.
     \return The constant \f$C_p=(p-1)/p\f$.
   */
  virtual double Cp(std::size_t const p) const final override;


};

} // namespace clf

#endif
