#include "clf/LinearDifferentialOperator.hpp"

using namespace clf;

LinearDifferentialOperator::LinearDifferentialOperator(Eigen::VectorXi const& counts, std::size_t const outdim) :
  indim(counts.size()), outdim(outdim),
  counts(std::vector<CountPair>(1, CountPair(counts, outdim)))
{}

LinearDifferentialOperator::LinearDifferentialOperator(std::vector<Eigen::VectorXi> const& counts) :
  indim(counts[0].size()), outdim(counts.size()),
  counts(ComputeCountPairs(counts))
{}

LinearDifferentialOperator::LinearDifferentialOperator(std::vector<CountPair> const& counts) :
  indim(counts[0].first.size()), outdim(ComputeOutputDimension(counts)),
  counts(counts)
{}

std::vector<LinearDifferentialOperator::CountPair> LinearDifferentialOperator::ComputeCountPairs(std::vector<Eigen::VectorXi> const& counts) {
  std::vector<CountPair> pairs(counts.size());
  for( std::size_t i=0; i<counts.size(); ++i ) {
    assert(counts[0].size()==counts[i].size());
    pairs[i] = CountPair(counts[i], 1);
  }
  return pairs;
}

std::size_t LinearDifferentialOperator::ComputeOutputDimension(std::vector<CountPair> const& counts) {
  std::size_t dim = 0;
  for( const auto& it : counts ) {
    assert(counts[0].first.size()==it.first.size());
    dim += it.second;
  }
  return dim;
}

LinearDifferentialOperator::CountPair LinearDifferentialOperator::Counts(std::size_t const ind) const {
  std::size_t jnd = 0;
  for( const auto& it : counts ) {
    jnd += it.second;
    if( jnd>ind ) { return it; }
  }

  assert(false);
  return CountPair();
}
