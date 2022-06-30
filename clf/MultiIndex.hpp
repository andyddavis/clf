#ifndef MULTIINDEX_HPP_
#define MULTIINDEX_HPP_

#include <vector>

namespace clf {

/// A multi-index \f$\alpha \in \mathbb{N}^{d}\f$
class MultiIndex {
public:

  /**
  @param[in] alpha The multi-index \f$\alpha \in \mathbb{N}^{d}\f$
  */
  MultiIndex(std::vector<std::size_t> const& alpha);

  virtual ~MultiIndex() = default;

  /// The dimension \f$d\f$ of the multi-index 
  std::size_t Dimension() const;

  /// Compute the order of the multi-index \f$\sum_{i=1}^{d} \alpha_i\f$.
  /**
     The order is the sum of the elements of alpha.
     \return The order \f$\sum_{i=1}^{d} \alpha_i\f$.
  */
  std::size_t Order() const;

  /// The multi-index \f$\alpha \in \mathbb{N}^{d}\f$
  const std::vector<std::size_t> alpha;

private:
};

} // namespace clf

#endif
