#include "clf/MultiIndex.hpp"

using namespace clf;

MultiIndex::MultiIndex(std::vector<std::size_t> const& alpha) :
  alpha(alpha) {}

std::size_t MultiIndex::Dimension() const { return alpha.size(); }

std::size_t MultiIndex::Order() const {
  std::size_t order = 0;
  for( const auto& it : alpha ) { order += it; }
  return order;
}
