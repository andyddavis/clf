#ifndef SINCOSBASIS_HPP_
#define SINCOSBASIS_HPP_

#include "clf/BasisFunctions.hpp"

namespace clf {

/// Represent a local function using a sine-cosine basis
/**
We need to define basis functions \f$\{\phi^{(i)}\}_{i=1}^{n}\f$ for the local function \f$\ell: \mathbb{R}^{d} \mapsto \mathbb{R}\f$.

Given the number \f$j \in \mathbb{N}\f$, define the scalar functions \f$l_i: \mathbb{R} \mapsto \mathbb{R}\f$
\f{equation*}{
    \begin{array}{cccccc}
        l_0(x) = 1 & l_{2j}(x) = \cos{(\pi j x)} & \mbox{(even numbers)} & \mbox{and} & l_{2j-1}(x) = \sin{(\pi j x)} & \mbox{(odd numbers.)}
    \end{array}
\f}
Let \f$\iota = (i_0,\, i_1,\, ...,\, i_d)\f$ define a multi-index such that the \f$i^{th}\f$ basis function \f$\phi^{(i)}: \mathbb{R}^{d} \mapsto \mathbb{R}\f$ is
\f{equation*}{
    \phi^{(i(\iota))} = \prod_{j=1}^{d} l_{i_j}(x_j).
\f}
For example, \f$\iota = (0,\, 0,\, ...,\, 0)\f$ defines the constant basis function \f$\phi^{(0)}(x) = 1\f$, \f$\iota = (1,\, 0,\, ...,\, 0)\f$ defines \f$\phi^{(i(\iota))}(x) = \sin{(\pi x_1)}\f$, and \f$\iota = (2,\, 0,\, ...,\, 0)\f$ defines \f$\phi^{(i(\iota))}(x) = \cos{(\pi x_1)}\f$, where we have implicitly defined a mapping between the index \f$i\f$ and the multi-index \f$\iota\f$.
*/
class SinCosBasis : public BasisFunctions {
public:

  /**
  @param[in] multis The multi-index set---each multi-index corresponds to a basis function
  @param[in] pt Options for the basis construction
  */
  SinCosBasis(std::shared_ptr<muq::Utilities::MultiIndexSet> const& multis, boost::property_tree::ptree const& pt);

  virtual ~SinCosBasis() = default;

  /// Construct a total-order sine/cosine basis
  /**
  For a multi-index \f$\iota\f$ define \f$\vert \iota \vert = \sum_{i=1}^{d} i_d\f$. This function constructs a polynomials basis using all of the multi-indices such that \f$\vert \iota \vert \leq 2 p\f$, where the order \f$p\f$ is a user-supplied parameter.

  Note we use all of the multi-indices whose order is bounded by \f$2 p\f$ so that we include all of the sine and cosine terms.

  <B>Configuration Parameters (in addition to the usual parameters for a PolynomialBasis):</B>
  Parameter Key | Type | Default Value | Description |
  ------------- | ------------- | ------------- | ------------- |
  "InputDimension"   | <tt>std::size_t</tt> | --- | The input dimension for the basis functions (the dimension \f$d\f$). |
  "Order"   | <tt>std::size_t</tt> | <tt>1</tt> | The order of the multi-indices (the parameter \f$p\f$). |
  @param[in] pt Options for the basis construction
  */
  static std::shared_ptr<SinCosBasis> TotalOrderBasis(boost::property_tree::ptree const& pt);

protected:

  /// Evaluate the scalar basis function \f$l_i: \mathbb{R} \mapsto \mathbb{R}\f$.
  /**
  Evaluate the scalar basis function
  \f{equation*}{
  \begin{array}{cccccc}
  l_0(x) = 1 & l_{2j}(x) = \cos{(\pi n x)} & \mbox{(even numbers)} & \mbox{and} & l_{2j-1}(x) = \sin{(\pi n x)} & \mbox{(odd numbers.)}
  \end{array}
  \f}
  @param[in] x The point where we are evaluating the scalar basis function
  @param[in] ind The index of the \f$i^{th}\f$ scalar basis function
  \return The scalar basis function evaluation
  */
  virtual double ScalarBasisFunction(double const x, std::size_t const ind) const override;

  /// Evaluate the \f$k^{th}\f$ derivative of the scalar basis function \f$\frac{d^k l_i}{d x^{k}}\f$.
  /**
  @param[in] x The point where we are evaluating the scalar basis function
  @param[in] ind The index of the \f$i^{th}\f$ scalar basis function
  @param[in] k We want the \f$k^{th}\f$ derivative
  \return The scalar basis function evaluation
  */
  virtual double ScalarBasisFunctionDerivative( double const x, std::size_t const ind, std::size_t const k) const override;

private:
};

} // namespace clf

#endif
