#ifndef POLYNOMIALBASIS_HPP_
#define POLYNOMIALBASIS_HPP_

#include <MUQ/Approximation/Polynomials/BasisExpansion.h>

#include "clf/BasisFunctions.hpp"

namespace clf {

/// Represent a local function using a polynomial basis
/**
Given the number \f$j \in \mathbb{N}\f$, define the scalar functions \f$l_i: \mathbb{R} \mapsto \mathbb{R}\f$ such that \f$l_j\f$ is a \f$j^{th}\f$ degree polynomial. Let \f$\iota = (i_0,\, i_1,\, ...,\, i_d)\f$ define a multi-index such that the \f$i^{th}\f$ basis function \f$\phi^{(i)}: \mathbb{R}^{d} \mapsto \mathbb{R}\f$ is
\f{equation*}{
    \phi^{(i(\iota))} = \prod_{j=1}^{d} l_{i_j}(x_j).
\f}

<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"ScalarBasis"   | <tt>std::string</tt> | <tt>"Legendre"</tt> | The orthogonal polynomial basis used for the scalar basis evaluation. |
*/
class PolynomialBasis : public BasisFunctions {
public:

  /**
  @param[in] multis The multi-index set---each multi-index corresponds to a basis function
  @param[in] pt Options for the basis construction
  */
  PolynomialBasis(std::shared_ptr<muq::Utilities::MultiIndexSet> const& multis, boost::property_tree::ptree const& pt);

  /// Construct a total-order polynomial basis
  /**
  For a multi-index \f$\iota\f$ define \f$\vert \iota \vert = \sum_{i=1}^{d} i_d\f$. This function constructs a polynomials basis using all of the multi-indices such that \f$\vert \iota \vert \leq p\f$, where the order \f$p\f$ is a user-supplied parameter.

  <B>Configuration Parameters (in addition to the usual parameters for a PolynomialBasis):</B>
  Parameter Key | Type | Default Value | Description |
  ------------- | ------------- | ------------- | ------------- |
  "InputDimension"   | <tt>std::size_t</tt> | --- | The input dimension for the basis functions (the dimension \f$d\f$). |
  "Order"   | <tt>std::size_t</tt> | <tt>2</tt> | The order of the multi-indices (the parameter \f$p\f$). |
  @param[in] pt Options for the basis construction
  */
  static std::shared_ptr<PolynomialBasis> TotalOrderBasis(boost::property_tree::ptree const& pt);

  virtual ~PolynomialBasis() = default;

protected:

  /// Evaluate the scalar basis function \f$l_i: \mathbb{R} \mapsto \mathbb{R}\f$.
  /**
  @param[in] ind The index of the \f$i^{th}\f$ scalar basis function
  @param[in] x The point where we are evaluating the scalar basis function
  \return The scalar basis function evaluation
  */
  virtual double ScalarBasisFunction(std::size_t const ind, double const x) const override;

private:

  /// The scalar basis we are using to construct the polynomial basis
  std::shared_ptr<muq::Approximation::IndexedScalarBasis> poly;
};

} // namespace clf

#endif
