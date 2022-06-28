#include "clf/MultiIndex.hpp"

using namespace clf;

MultiIndex::MultiIndex(std::vector<std::size_t> const& indices) :
  indices(indices) {}

std::size_t MultiIndex::Dimension() const { return indices.size(); }
