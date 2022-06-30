#ifndef FEATUREMATRIX_HPP_
#define FEATUREMATRIX_HPP_

#include "clf/FeatureVector.hpp"

namespace clf {

/// A matrix of basis function \f$\Phi: \mathcal{B} \mapsto \mathbb{R}^{\bar{q} \times m}\f$.
/**
Let \f$\{ \phi_i: \mathcal{B} \mapsto \mathbb{R}^{q_i} \}_{i=1}^{m}\f$ be feature vectors (see clf::FeatureVector). Define the feature matrix 
\f{equation*}{
   \Phi(x) = \begin{bmatrix}
   \phi_1(x) & 0 & 0 & ... & 0 \\ \hline
   0 & \phi_2(x) & 0 & ... & 0 \\ \hline
   \vdots & ... & \vdots & \ddots & \vdots \\ \hline
   0 & 0 & ... & 0 & \phi_m(x) \\ 
   \end{bmatrix} \in \mathbb{R}^{\bar{q} \times m},
\f}
where \f$\bar{q} = \sum_{i=1}^{m} q_i\f$.
*/
class FeatureMatrix {
public:

  /// Each vector pair is a feature vector \f$\phi\f$ and the number of times it is evaluated in the feature matrix
  typedef std::pair<std::shared_ptr<const FeatureVector>, std::size_t> VectorPair;

  /// Construct a feature matrix with \f$m\f$ feature vectors using the same feature vector for all \f$\phi_i\f$.
  /**
     @param[in] vec A feature vector \f$\phi: \mathcal{B} \mapsto \mathbb{R}^{q}\f$. All \f$\phi_i\f$ will be \f$\phi\f$.
     @param[in] numFeatureVectors The number of feature vectors \f$m\f$
  */
  FeatureMatrix(std::shared_ptr<const FeatureVector> const& vec, std::size_t const numFeatureVectors);

  /// Construct a feature matrix with \f$m\f$ feature vectors
  /**
     We are given a vector of feature vectors that define the feature matrix. Each feature vector \f$\phi_j\f$ is repeated multiple times; the number of times is the second entry in each pair.
     @param[in] featureVectors The feature vectors and the number of times they are repeated 
   */
  FeatureMatrix(std::vector<VectorPair> const& featureVectors);

  /// Construct a feature matrix with \f$m\f$ feature vectors
  /**
     We are given a vector of feature vectors that define the feature matrix.
     @param[in] vecs The feature vectors 
   */
  FeatureMatrix(std::vector<std::shared_ptr<const FeatureVector> > const& vecs);

  virtual ~FeatureMatrix() = default;

  /// Apply the feature matrix transpose to coefficients \f$\Phi(x)^{\top} c\f$
  /**
     @param[in] x The point \f$x\f$ where we are evaluating the feature matrix 
     @param[in] coeff The coefficients \f$c\f$
     \return The feature matrix transpose applied to the coefficients \f$\Phi(x)^{\top} c\f$
  */
  Eigen::VectorXd ApplyTranspose(Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const;

  /// The total number of basis function \f$\bar{q}\f$
  const std::size_t numBasisFunctions;

  /// The number of feature vectors \f$m\f$
  const std::size_t numFeatureVectors;

private:

  /// Compute the number of basis functions given a list of feature vectors 
  /**
     @param[in] vecs A list of feature vectors 
     \return The number of basis functions
   */
  static std::size_t ComputeNumBasisFunctions(std::vector<std::shared_ptr<const FeatureVector> > const& vecs);
  
  /// Compute a vector of FeatureMatrix::VectorPairs given a vector of feature vectors that are only repeated once
  /**
     @param[in] vecs A list of feature vectors 
     \return A vector of FeatureMatrix::VectorPairs 
   */
  static std::vector<VectorPair> CreateVectorPairs(std::vector<std::shared_ptr<const FeatureVector> > const& vecs);

  /// Compute the number of basis functions given a list of feature vectors and the number of times they are evaluated
  /**
     @param[in] featureVectors A list of feature vectors and the number of times they are evaluated 
     \return The number of basis functions
   */
  static std::size_t ComputeNumBasisFunctions(std::vector<VectorPair> const& featureVectors);

  /// Compute the number of feature vectors given a list of feature vectors and the number of times they are evaluated
  /**
     @param[in] featureVectors A list of feature vectors and the number of times they are evaluated 
     \return The number of feature vectors
   */
  static std::size_t ComputeNumFeatureVectors(std::vector<VectorPair> const& featureVectors);

  /// A vector of the feature vectors \f$\phi_i\f$
  /**
     Each pair corresponds to a feature vector \f$\phi_i\f$ and the number of times it is repeated in the feature matrix.
   */
  const std::vector<VectorPair> featureVectors;
};

} // namespace clf 

#endif
