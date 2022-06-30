#ifndef MULTIINDEXSET_HPP_
#define MULTIINDEXSET_HPP_

#include <memory>

#include "clf/MultiIndex.hpp"

namespace clf {

/// A set of multi-indices \f$\{ \alpha_i \in \mathbb{R}^{d} \}_{i=1}^{q}\f$
/**
Each multi-index \f$\alpha_i \in \mathbb{N}^{d}\f$ is a clf::MultiIndex.
*/
class MultiIndexSet {
public:

  /**
  @param[in,out] inds A list of multi-indices \f$\{ \alpha_i \in \mathbb{R}^{d} \}_{i=1}^{q}\f$, after construction the vector inds contains invalid pointers (this object takes ownership of the multi indices)
  */
  MultiIndexSet(std::vector<std::unique_ptr<MultiIndex> >& inds);

  virtual ~MultiIndexSet() = default;

  /// Create a total-order index set 
  static std::unique_ptr<MultiIndexSet> CreateTotalOrder(std::size_t const dim, std::size_t const maxOrder);

  /// The dimension \f$d\f$ of each multi-index 
  std::size_t Dimension() const;

  /// The number of multi-indices \f$q\f$
  std::size_t NumIndices() const;

  /// The maximum index of the \f$j^{\text{th}}\f$ component \f$\max_{i \in \{1,\,...,\,q\}} \alpha_{i,j} \f$ (for \f$j \in \{1,\, ...,\, d\}\f$).
  std::size_t MaxIndex(std::size_t const j) const;

  /// A list of multi-indices \f$\{ \alpha_i \in \mathbb{R}^{d} \}_{i=1}^{q}\f$
  const std::vector<std::unique_ptr<MultiIndex> > indices;

private:

  /// Create a total-order index set 
  static void CreateTotalOrder(std::size_t const maxOrder, std::size_t const currDim, std::vector<std::size_t>& base, std::vector<std::unique_ptr<MultiIndex> >& indices);

  /// The maximum index for each dimension. The \f$j^{\text{th}}\f$ component is \f$\max_{i \in \{1,\,...,\,q\}} \alpha_{i,j} \f$ (for \f$j \in \{1,\, ...,\, d\}\f$).
  std::vector<std::size_t> maxIndices;
};

} // namespace clf

#endif
