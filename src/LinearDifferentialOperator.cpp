#include "clf/LinearDifferentialOperator.hpp"

using namespace clf;

LinearDifferentialOperator::LinearDifferentialOperator(Eigen::MatrixXi const& counts, std::size_t const outdim) :
  indim(counts.rows()), outdim(outdim),
  counts(std::vector<CountPair>(1, CountPair(counts, outdim)))
{}

LinearDifferentialOperator::LinearDifferentialOperator(std::vector<Eigen::MatrixXi> const& counts) :
  indim(counts[0].rows()), outdim(counts.size()),
  counts(ComputeCountPairs(counts))
{}

LinearDifferentialOperator::LinearDifferentialOperator(std::vector<CountPair> const& counts) :
  indim(counts[0].first.rows()), outdim(ComputeOutputDimension(counts)),
  counts(counts)
{}

std::size_t LinearDifferentialOperator::NumOperators() const { return counts[0].first.cols(); }

std::vector<LinearDifferentialOperator::CountPair> LinearDifferentialOperator::ComputeCountPairs(std::vector<Eigen::MatrixXi> const& counts) {
  std::vector<CountPair> pairs(counts.size());
  for( std::size_t i=0; i<counts.size(); ++i ) {
    assert(counts[0].cols()==counts[i].cols());
    assert(counts[0].rows()==counts[i].rows());
    pairs[i] = CountPair(counts[i], 1);
  }
  return pairs;
}

std::size_t LinearDifferentialOperator::ComputeOutputDimension(std::vector<CountPair> const& counts) {
  std::size_t dim = 0;
  for( const auto& it : counts ) {
    assert(counts[0].first.cols()==it.first.cols());
    assert(counts[0].first.rows()==it.first.rows());
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
