#ifndef LOCALFUNCTION_HPP_
#define LOCALFUNCTION_HPP_

#include "clf/FeatureMatrix.hpp"

namespace clf {

/// A function \f$u: \mathcal{B} \mapsto \mathbb{R}^{m}\f$, where \f$\mathcal{B} \subseteq \mathbb{R}^{d}\f$
/**
The local function \f$u: \mathcal{B} \mapsto \mathbb{R}^{m}\f$ is defined by the feature matrix (see clf::FeatureMatrix)
\f{equation*}{
   \Phi(x) = \begin{bmatrix}
   \phi_1(x) & 0 & 0 & ... & 0 \\ \hline
   0 & \phi_2(x) & 0 & ... & 0 \\ \hline
   \vdots & ... & \vdots & \ddots & \vdots \\ \hline
   0 & 0 & ... & 0 & \phi_m(x) \\ 
   \end{bmatrix} \in \mathbb{R}^{\bar{q} \times m}
\f}
and coefficients \f$c \in \mathbb{R}^{\bar{q}}\f$ such that \f$u(x) = \Phi(x)^{\top} c\f$ for \f$x \in \mathcal{B}\f$.
*/
class LocalFunction {
public:

  /// Construct a local function
  /**
     @param[in] featureMatrix The feature matrix that defines this local function
   */
  LocalFunction(std::shared_ptr<const FeatureMatrix> const& featureMatrix);

  virtual ~LocalFunction() = default;

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

  /// The number of coefficients \f$\bar{q}\f$
  /**
     \return The number of coefficients \f$\bar{q}\f$
  */
  std::size_t NumCoefficients() const;

  /// Evaluate the location function \f$u(x; c)\f$ at a point \f$x \in \mathcal{B}\f$ given coefficients \f$c \in \mathbb{R}^{\bar{q}}\f$
  /**
     @param[in] x The location where we want to evaluate the location function 
     @param[in] coeff The coefficients \f$c \in \mathbb{R}^{\bar{q}}\f$ that define this location function 
     \return The local function evaluation \f$u(x)\f$
   */
  Eigen::VectorXd Evaluate(Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const;

private:

  /// The feature matrix \f$\Phi(x)\f$ that defines this local function
  std::shared_ptr<const FeatureMatrix> featureMatrix;
};

} // namespace clf

#endif
