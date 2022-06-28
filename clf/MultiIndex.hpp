#ifndef MULTIINDEX_HPP_
#define MULTIINDEX_HPP_

#include <vector>

namespace clf {

/// A multi-index \f$\boldsymbol{\alpha} \in \mathbb{N}^{d}\f$
class MultiIndex {
public:

  MultiIndex(std::vector<std::size_t> const& indices);

  virtual ~MultiIndex() = default;

  /// The dimension \f$d\f$ of the multi-index 
  std::size_t Dimension() const;

  /// The multi-index \f$\boldsymbol{\alpha} \in \mathbb{N}^{d}\f$
  const std::vector<std::size_t> indices;

private:
};

} // namespace clf

#endif
