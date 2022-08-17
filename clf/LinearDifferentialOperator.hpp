#ifndef LINEARDIFFERENTIALOPERATOR_HPP_
#define LINEARDIFFERENTIALOPERATOR_HPP_

#include <Eigen/Core>

namespace clf {

/// Define a linear differential operator
/**
The \f$i^{\text{th}}\f$ component of the linear differential operator is defined by 
\f{equation*}{
   L_i = \frac{\partial^{\vert B_i \vert}}{\prod_{j=1}^{\vert B_i \vert} \partial x_{B_{i,j}}}
\f}

Applying the differential operator to a function \f$u: \Omega \mapsto \mathbb{R}^{m}\f$ (see clf::LocalFunction) computes 
\f{equation*}{
\begin{bmatrix}
\frac{\partial^{\vert B_0 \vert}}{\prod_{j=1}^{\vert B_0 \vert} \partial x_{B_{0,j}}} u_0 \\
\vdots \\
\frac{\partial^{\vert B_m \vert}}{\prod_{j=1}^{\vert B_m \vert} \partial x_{B_{m,j}}} u_m
\end{bmatrix}
\f}
*/
class LinearDifferentialOperator {
public:

  /// First: The number of components this operator is applied to, Second: Each columin has length equal to the input dimension and the \f$i^{\text{th}}\f$ component is the number of times we are taking the derivative with respect to the \f$i^{\text{th}}\f$ input. Each column corresponds to a different differential operator.
  typedef std::pair<Eigen::MatrixXi, std::size_t> CountPair;

  /// Apply the same linear differential operator too all components
  /**
     @param[in] counts A vector whose length is the input dimension and the \f$i^{\text{th}}\f$ component is the number of times we are taking the derivative with respect to the \f$i^{\text{th}}\f$ input.
     @param[in] outdim The number of components of the linear differential operator
   */
  LinearDifferentialOperator(Eigen::MatrixXi const& counts, std::size_t const outdim);

  /// Apply the a different linear differential operator to all components
  /**
     @param[in] counts The counts for each component
   */
  LinearDifferentialOperator(std::vector<Eigen::MatrixXi> const& counts);

  /// Apply the a different linear differential operator to some components
  /**
     @param[in] counts The counts for each component
   */
  LinearDifferentialOperator(std::vector<CountPair> const& counts);

  virtual ~LinearDifferentialOperator() = default;

  /// The number of differential operators defined by this object
  /**
     \return The number of columns in each <tt>first</tt> of clf::LinearDifferentialOperator. 
   */
  std::size_t NumOperators() const;

  ///  A vector whose length is the input dimension and the \f$i^{\text{th}}\f$ component is the number of times we are taking the derivative with respect to the \f$i^{\text{th}}\f$ input.
  /**
     @param[in] ind We want the differential operator applied to the \f$i^{th}\f$
   */
  CountPair Counts(std::size_t const ind) const;

  /// The number of input components
  const std::size_t indim;

  /// The dimension of the linear operator
  const std::size_t outdim;
private:

  /// Compute the LinearDifferentialOperator::CountPair given that we are applying a different operator to each component
  /**
     @param[in] counts The counts for each component
     \return The count pairs for each component
   */
  static std::vector<CountPair> ComputeCountPairs(std::vector<Eigen::MatrixXi> const& counts);

  /// Compute the output dimension
  /**
     @param[in] counts The counts for each component
     \return The output dimension
   */
  static std::size_t ComputeOutputDimension(std::vector<CountPair> const& counts);

  /// The counds for the number of times we are differenting with respect to each input component
  /**
     First: The number of components this operator applies to, Second: The differential operator for each component
   */
  const std::vector<CountPair> counts;
};
  
} // namespace clf

#endif
