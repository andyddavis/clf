#ifndef SUPPORTPOINT_HPP_
#define SUPPORTPOINT_HPP_

#include <boost/property_tree/ptree.hpp>

#include <MUQ/Modeling/ModPiece.h>

#include "clf/SupportPointExceptions.hpp"
#include "clf/BasisFunctions.hpp"

namespace clf {

/// The local function \f$\ell\f$ associated with a support point \f$x\f$.
/**
Let \f$x \in \Omega \subseteq \mathbb{R}^{d}\f$ be a support point with an associated local function \f$\ell: \Omega \mapsto \mathbb{R}^{m}\f$. Suppose that we are building an approximation of the function \f$u:\Omega \mapsto \mathbb{R}^{m}\f$. Although \f$\ell\f$ is well-defined in the entire domain, we expect there is a smooth monotonic function \f$W: \Omega \mapsto \mathbb{R}^{+}\f$ with \f$W(0) \leq \epsilon\f$ and \f$W(r) \rightarrow \infty\f$ as \f$r \rightarrow \infty\f$ such that \f$\| \ell(y) - u(x) \|^2 \leq W(\|y-x\|^2)\f$. Therefore, we primarily care about the local function \f$\ell\f$ in a ball \f$\mathcal{B}_{\delta}(x)\f$ centered at \f$x\f$ with radius \f$\delta\f$.

Define the local coordinate \f$\hat{x}(y) = (y-x)/\delta\f$ (parameterized by \f$\delta\f$) and the basis functions
\f{equation*}{
    \phi(y) = [\phi_1(\hat{x}(y)),\, \phi_2(\hat{x}(y)),\, ...,\, \phi_q(\hat{x}(y))]^{\top}.
\f}

<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"OutputDimension"   | <tt>std::size_t</tt> | <tt>1</tt> | The output dimension of the support point. |
"InitialRadius"   | <tt>double</tt> | <tt>1.0</tt> | The initial value of the \f$\delta\f$ parameter. |
*/
class SupportPoint : public muq::Modeling::ModPiece {
public:

  /**
  @param[in] x The location of the support point \f$x\f$
  @param[in] pt The options for the support point
  */
  SupportPoint(Eigen::VectorXd const& x, boost::property_tree::ptree const& pt);

  virtual ~SupportPoint() = default;

  /// The input dimension \f$d\f$
  /**
  \return The input dimension \f$d\f$
  */
  std::size_t InputDimension() const;

  /// The output dimension \f$m\f$
  /**
  \return The output dimension \f$m\f$
  */
  std::size_t OutputDimension() const;

  /// The radius of the ball the defines where the local function is relatively accurate
  /**
  \return The radius of the ball the defines where the local function is relatively accurate
  */
  double Radius() const;

  /// The radius of the ball the defines where the local function is relatively accurate
  /**
  \return The radius of the ball the defines where the local function is relatively accurate
  */
  double& Radius();

  /// Transform into the local coordinate
  /**
  Given the global coordinate \f$y \in \Omega\f$ compute \f$\hat{x}(y) = (y-x)/\delta\f$.
  @param[in] y The global coordinate \f$y \in \Omega\f$
  \return The local coordinate \f$\hat{x}(y) = (y-x)/\delta\f$
  */
  Eigen::VectorXd LocalCoordinate(Eigen::VectorXd const& y) const;

/// Transform into the global coordinate
  /**
  Given the local coordinate \f$\hat{x} \in \mathbb{R}^{d}\f$ compute \f$y = \delta \hat{x} + x\f$.
  @param[in] y The local coordinate \f$\hat{x} \in \mathbb{R}^{d}\f$
  \return The global coordinate \f$y = \delta \hat{x} + x\f$
  */
  Eigen::VectorXd GlobalCoordinate(Eigen::VectorXd const& xhat) const;

  /// The location of the support point \f$x\f$.
  const Eigen::VectorXd x;

private:

  /// Create the basis functions from the given options
  /**
  @param[in] indim The input dimension for the support point
  @param[in] pt The options for the basis functions
  */
  static std::shared_ptr<BasisFunctions> CreateBasisFunctions(std::size_t const indim, boost::property_tree::ptree pt);

  /// Evaluate the local function \f$\ell\f$ associated with this support point
  /**
  Fills in the <tt>outputs</tt> vector attached to <tt>this</tt> SupportPoint (inherited from <tt>muq::Modeling::ModPiece</tt>). This is a vector of length <tt>1</tt> that stores the local function evaluation.
  @param[in] inputs There is only one input and it is the evaluation point
  */
  virtual void EvaluateImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& input) override;

  /// The basis that defines this support point
  std::shared_ptr<BasisFunctions> basis;

  /// The parameter \f$\delta\f$ that defines the radius for which we expect this local function to be relatively accurate
  /**
  The default value is \f$\delta=1\f$. This parameter defines the local coordinate transformation \f$\hat{x}(y) = (y-x)/\delta\f$.
  */
  double delta;
};

} // namespace clf

#endif
