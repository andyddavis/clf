#ifndef FEATUREVECTOR_HPP_
#define FEATUREVECTOR_HPP_

#include "clf/MultiIndexSet.hpp"

#include "clf/BasisFunctions.hpp"

namespace clf {

/// A vector of basis function \f$\phi: \mathcal{B} \mapsto \mathbb{R}^q\f$, where \f$\mathcal{B} \subset \mathbb{R}^{d}\f$ is a compact domain.
/**
Let \f$\{ \alpha_i \in \mathbb{N}^{d} \}_{i=1}^{q}\f$ be a set of multi-index (see clf::MultiIndexSet) and \f$\varphi_{\alpha_{i,j}}: \mathcal{D}_j \mapsto \mathbb{R}\f$ be a basis function (see clf::BasisFunctions), where \f$\mathcal{B} = \mathcal{D}_0 \times ... \times \mathcal{D}_d\f$. The feature vector is 
\f{equation*}{
   \phi(x) = \begin{bmatrix}
   \prod_{j=1}^{d} \varphi_{\alpha_{0,j}}(x_j) \	\
   \vdots \						\
   \prod_{j=1}^{d} \varphi_{\alpha_{q,j}}(x_j) 
   \end{bmatrix}.
\f}
*/
class FeatureVector {
public:

  /**
     @param[in] set The multi-index set that defines this feature vector
     @param[in] basis The basis functions used to define this feature vector
  */
  FeatureVector(std::shared_ptr<const MultiIndexSet> const& set, std::shared_ptr<BasisFunctions> const& basis);

  virtual ~FeatureVector() = default;

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

private:

  /// The multi-index set that defines this feature vector
  std::shared_ptr<const MultiIndexSet> set;

  /// The basis functions used to define this feature vector
  std::shared_ptr<const BasisFunctions> basis;
};

};

#endif
