#ifndef MULTIINDEXSET_HPP_
#define MULTIINDEXSET_HPP_

#include <memory>

#include "clf/Parameters.hpp"

#include "clf/MultiIndex.hpp"

namespace clf {

/// A set of multi-indices \f$\{ \alpha_i \in \mathbb{R}^{d} \}_{i=1}^{q}\f$
/**
Each multi-index \f$\alpha_i \in \mathbb{N}^{d}\f$ is a clf::MultiIndex.
*/
class MultiIndexSet {
public:

  /**
  @param[in,out] inds A list of multi-indices \f$\{ \alpha_i \in \mathbb{R}^{d} \}_{i=1}^{q}\f$, after construction the vector inds contains invalid objects (this object takes ownership of the multi indices)
  */
  MultiIndexSet(std::vector<MultiIndex>& inds);

  virtual ~MultiIndexSet() = default;

  /// Create a total-order index set 
  /**
     @param[in] dim The dimension of the multi-index
     @param[in] maxOrder The maximum order
   */
  static std::unique_ptr<MultiIndexSet> CreateTotalOrder(std::size_t const dim, std::size_t const maxOrder);

  /// Create a total-order index set 
  /**
     <B>Configuration Parameters:</B>
     Parameter Key | Type | Default Value | Description |
     ------------- | ------------- | ------------- | ------------- |
     "InputDimension"   | <tt>std::size_t</tt> | --- | The dimension of the multi-index. This is a required parameter. |
     "MaximumOrder"   | <tt>std::size_t</tt> | --- | The maximum order of the multi-index. This is a required parameter. |
     @param[in] para The parameters
   */
  static std::unique_ptr<MultiIndexSet> CreateTotalOrder(std::shared_ptr<Parameters> const& para);

  /// The dimension \f$d\f$ of each multi-index 
  std::size_t Dimension() const;

  /// The number of multi-indices \f$q\f$
  std::size_t NumIndices() const;

  /// The maximum index of the \f$j^{\text{th}}\f$ component \f$\max_{i \in \{1,\,...,\,q\}} \alpha_{i,j} \f$ (for \f$j \in \{1,\, ...,\, d\}\f$).
  std::size_t MaxIndex(std::size_t const j) const;

  /// A list of multi-indices \f$\{ \alpha_i \in \mathbb{R}^{d} \}_{i=1}^{q}\f$
  const std::vector<MultiIndex> indices;

private:

  /// Create a total-order index set 
  /**
     @param[in] maxOrder The maximum order
     @param[in] currDim The current dimension that we are adding 
     @param[in] base The basis multi-index 
     @param[in,out] indices The indices we have added so far and are appending to
   */
  static void CreateTotalOrder(std::size_t const maxOrder, std::size_t const currDim, std::vector<std::size_t>& base, std::vector<MultiIndex>& indices);

  /// The maximum index for each dimension. The \f$j^{\text{th}}\f$ component is \f$\max_{i \in \{1,\,...,\,q\}} \alpha_{i,j} \f$ (for \f$j \in \{1,\, ...,\, d\}\f$).
  std::vector<std::size_t> maxIndices;
};

} // namespace clf

#endif
