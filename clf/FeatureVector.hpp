#ifndef FEATUREVECTOR_HPP_
#define FEATUREVECTOR_HPP_

#include "clf/Domain.hpp"

#include "clf/Point.hpp"

#include "clf/MultiIndexSet.hpp"

#include "clf/BasisFunctions.hpp"

namespace clf {

/// A vector of basis function \f$\phi: \mathcal{B} \mapsto \mathbb{R}^q\f$, where \f$\mathcal{B} \subseteq \mathbb{R}^{d}\f$ 
/**
Let \f$\{ \alpha_i \in \mathbb{N}^{d} \}_{i=1}^{q}\f$ be a set of multi-index (see clf::MultiIndexSet) and \f$\varphi_{\alpha_{i,j}}: \mathcal{D}_j \mapsto \mathbb{R}\f$ be a basis function (see clf::BasisFunctions), where \f$\mathcal{B} = \mathcal{D}_0 \times ... \times \mathcal{D}_d\f$. The feature vector is 
\f{equation*}{
   \phi(F(x)) = \phi(y) = \begin{bmatrix}
   \prod_{j=1}^{d} \varphi_{\alpha_{0,j}}(y_j) \\
   \vdots \\
   \prod_{j=1}^{d} \varphi_{\alpha_{q,j}}(y_j) 
   \end{bmatrix},
\f}
where \f$y = F(x) = \delta^{-1} (x-\bar{x})\f$ is a linear transformation.
*/
class FeatureVector {
public:

  /**
     @param[in] set The multi-index set that defines this feature vector
     @param[in] basis The basis functions used to define this feature vector
     @param[in] domain The local domain of the feature vector
  */
  FeatureVector(std::shared_ptr<const MultiIndexSet> const& set, std::shared_ptr<BasisFunctions> const& basis, std::shared_ptr<Domain> const& domain = nullptr);

  virtual ~FeatureVector() = default;

  /// The input dimension \f$d\f$ 
  /**
     \return The input dimension \f$d\f$ 
   */
  std::size_t InputDimension() const;

  /// The number of basis functions \f$q\f$
  /**
     \return The number of basis functions \f$q\f$
   */
  std::size_t NumBasisFunctions() const;

  /// Evaluate the basis vector 
  /**
     @param[in] x The point where we want to evaluate the basis vector 
     \return The basis vector \f$\phi(x)\f$
   */
  Eigen::VectorXd Evaluate(Eigen::VectorXd const& x) const;

  /// Evaluate the basis vector derivative
  /**
     \f{equation*}{
     \frac{\partial^{\vert B \vert}}{\prod_{i=1}^{\vert B \vert} \partial x_{B_i}} \phi(F(x)) = \frac{\partial^{\vert B \vert}}{\prod_{i=1}^{\vert B \vert} \partial x_{B_i}} \begin{bmatrix}
     \phi_{0}(F(x)) \\
     \vdots \\
     \phi_{q}(F(x)) 
     \end{bmatrix} = \frac{\partial^{\vert B \vert}}{\prod_{i=1}^{\vert B \vert} \partial x_{B_i}} \begin{bmatrix}
     \prod_{j=1}^{d} \varphi_{\alpha_{0,j}}(F_j(x)) \\
     \vdots \\
     \prod_{j=1}^{d} \varphi_{\alpha_{q,j}}(F_j(x)) 
     \end{bmatrix},
     \f}
     @param[in] x The point where we want to evaluate the basis vector 
     @param[in] counts The number of times we are differentiating with respect to each index
     \return The derivative of the basis vector \f$\frac{\partial^{\vert B \vert}}{\prod_{i=1}^{\vert B \vert} \partial x_{B_i}} \phi(F(x))\f$
   */
  Eigen::VectorXd Derivative(Eigen::VectorXd const& x, Eigen::VectorXi const& counts) const;

  /// Evaluate the basis vector derivative
  /**
     \f{equation*}{
     \frac{\partial^{\vert B \vert}}{\prod_{i=1}^{\vert B \vert} \partial x_{B_i}} \phi(F(x)) = \frac{\partial^{\vert B \vert}}{\prod_{i=1}^{\vert B \vert} \partial x_{B_i}} \begin{bmatrix}
     \phi_{0}(F(x)) \\
     \vdots \\
     \phi_{q}(F(x)) 
     \end{bmatrix} = \frac{\partial^{\vert B \vert}}{\prod_{i=1}^{\vert B \vert} \partial x_{B_i}} \begin{bmatrix}
     \prod_{j=1}^{d} \varphi_{\alpha_{0,j}}(F_j(x)) \\
     \vdots \\
     \prod_{j=1}^{d} \varphi_{\alpha_{q,j}}(F_j(x)) 
     \end{bmatrix},
     \f}
     @param[in] x The point where we want to evaluate the basis vector 
     @param[in] counts The number of times we are differentiating with respect to each index
     @param[in] jac The jacobian of the coordinate transformation
     \return The derivative of the basis vector \f$\frac{\partial^{\vert B \vert}}{\prod_{i=1}^{\vert B \vert} \partial x_{B_i}} \phi(F(x))\f$
   */
  Eigen::VectorXd Derivative(Eigen::VectorXd const& x, Eigen::VectorXi const& counts, std::optional<Eigen::VectorXd> const& jac) const;
private:

  /// Evaluate the basis functions \f$\varphi_j(y_i)\f$
  /**
     @param[in] x The point where we want to evaluate the basis vector 
     \return The \f$i^{\text{th}}\f$ entry is a vector such that the \f$j^{\text{th}}\f$ component is \f$\varphi_j(y_i)\f$ for \f$j \in \{0,\, ...,\, \max{(\alpha_{:,j})} \}\f$
   */
  std::vector<Eigen::VectorXd> BasisEvaluation(Eigen::VectorXd const& x) const;

  /// Evaluate the basis function derivatives \f$\frac{\partial^{k}}{\partial x^{k}} \varphi_j(y_i)\f$
  /**
     @param[in] x The point where we want to evaluate the basis vector 
     @param[in] count The maximum derivative order for each dimension
     \return The \f$i^{\text{th}}\f$ entry is a vector such that the \f$(j, k)^{\text{th}}\f$ component is \f$\frac{\partial^k}{\partial y_j^k}\varphi_j(y_i)\f$ for \f$j \in \{0,\, ...,\, \max{(\alpha_{:,j})} \}\f$ and \f$k \in \{1,\, ...,\, K_d\}\f$
   */
  std::vector<Eigen::MatrixXd> BasisDerivatives(Eigen::VectorXd const& x, Eigen::VectorXi const& count) const;

  /// The multi-index set that defines this feature vector
  std::shared_ptr<const MultiIndexSet> set;

  /// The basis functions used to define this feature vector
  std::shared_ptr<const BasisFunctions> basis;

  /// The domain where this feature vector is defined
  std::shared_ptr<const Domain> domain;
};

} // namespace clf

#endif
