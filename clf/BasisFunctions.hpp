#ifndef BASISFUNCTIONS_HPP_
#define BASISFUNCTIONS_HPP_

#include <map>
#include <iostream>

#include <boost/property_tree/ptree.hpp>

#include <MUQ/Utilities/RegisterClassName.h>
#include <MUQ/Utilities/MultiIndices/MultiIndexSet.h>

#include "clf/BasisFunctionsExceptions.hpp"

namespace clf {

/// Evaluate the basis functions \f$\phi(x) = [\phi_1(x),\, \phi_2(x),\, ...,\, \phi_q(x)]\f$
/**
We need basis functions \f$\{\phi^{(i)}: \mathbb{R}^{d} \mapsto \mathbb{R} \}_{i=1}^{q}\f$ that define the local function \f$\ell: \mathbb{R}^{d} \mapsto \mathbb{R}\f$. We evaluate the local function \f$\ell(x) = c \cdot \phi(x)\f$, where \f$c \in \mathbb{R}^{q}\f$ are basis coefficients and \f$\phi(x) = [\phi_1(x),\, \phi_2(x),\, ...,\, \phi_q(x)]\f$.

In general, we define the basis functions by frist defining scalar functions \f$l_i: \mathbb{R} \mapsto \mathbb{R}\f$. Let \f$\iota = (i_0,\, i_1,\, ...,\, i_d)\f$ define a multi-index. The \f$i^{th}\f$ basis function \f$\phi^{(i)}: \mathbb{R}^{d} \mapsto \mathbb{R}\f$ is
\f{equation*}{
    \phi^{(i(\iota))}(x) = \prod_{j=1}^{d} l_{i_j}(x_j),
\f}
where we have implicitly defined a one-to-one mapping between the index \f$i\f$ and the multi-index \f$\iota\f$.
*/
class BasisFunctions {
public:
  /**
  @param[in] multis The multi-index set---each multi-index corresponds to a basis function
  @param[in] pt The options for the basis functions
  */
  BasisFunctions(std::shared_ptr<muq::Utilities::MultiIndexSet> const& multis, boost::property_tree::ptree const& pt);

  /// A constructor type for BasisFunctions, calling this type of function creates a BasisFunctions based on the options in the <tt>ptree</tt>.
  typedef std::function<std::shared_ptr<BasisFunctions>(std::shared_ptr<muq::Utilities::MultiIndexSet> const&, boost::property_tree::ptree const&)> BasisFunctionsConstructor;

  /// A map from the basis function name (a child of BasisFunctions) to the constructor that creates it
  typedef std::map<std::string, BasisFunctionsConstructor> BasisFunctionsMap;

  /// Get the static map from the basis function name to the constuctor that creates it
  /**
  \return The static map from the basis function name to the constuctor that creates it
  */
  static std::shared_ptr<BasisFunctionsMap> GetBasisFunctionsMap();

  /// Construct a basis function given different options
  /**
  The type of basis function to create should be stored in a string labeled <tt>"Type"</tt> in the <tt>ptree</tt>
  @param[in] multis The multi-index set---each multi-index corresponds to a basis function
  @param[in] pt Options for the basis function
  \return The basis function
  */
  static std::shared_ptr<BasisFunctions> Construct(std::shared_ptr<muq::Utilities::MultiIndexSet> const& multis, boost::property_tree::ptree const& pt);

  virtual ~BasisFunctions() = default;

  /// Evaluate the basis functions
  /**
  The vector of basis functions is \f$\phi(x) = [\phi_1(x),\, \phi_2(x),\, ...,\, \phi_q(x)]\f$.
  @param[in] x The point where we are evaluating the basis function
  \return The vector of basis function evaluations
  */
  Eigen::VectorXd EvaluateBasisFunctions(Eigen::VectorXd const& x) const;

  /// Evaluate the \f$i^{th}\f$ basis function
  /**
  Let \f$\iota(i)\f$ be the corresponding multi-index. The basis function is
  \f{equation*}{
  \phi^{(i(\iota))}(x) = \prod_{j=1}^{d} l_{i_j}(x_j),
  \f}
  where \f$l_{i}\f$ are BasisFunctions::ScalarBasisFunction evaluations.
  @param[in] ind The index of the basis function we are evaluating
  @param[in] x The point where we are evaluating the basis function
  \return The basis function evaluation
  */
  double EvaluateBasisFunction(std::size_t const ind, Eigen::VectorXd const& x) const;

  /// The number of basis functions
  /**
  \return The number of basis functions
  */
  std::size_t NumBasisFunctions() const;

protected:

  /// Evaluate the scalar basis function \f$l_i: \mathbb{R} \mapsto \mathbb{R}\f$.
  /**
  The basis function is defined by the product of these scalar functions. This must be implemented by a child.
  @param[in] ind The index of the \f$i^{th}\f$ scalar basis function
  @param[in] x The point where we are evaluating the scalar basis function
  \return The scalar basis function evaluation
  */
  virtual double ScalarBasisFunction(std::size_t const ind, double const x) const = 0;

private:
  /// The multi-index set---each multi-index corresponds to a basis function
  std::shared_ptr<muq::Utilities::MultiIndexSet> multis;
};

} // namespace clf

#define CLF_REGISTER_BASIS_FUNCTION(NAME) static auto reg ##NAME = clf::BasisFunctions::GetBasisFunctionsMap()->insert(std::make_pair(#NAME, muq::Utilities::shared_factory<NAME>()));

#endif
