#include "clf/MultiIndexSet.hpp"

#include <assert.h>

using namespace clf;

MultiIndexSet::MultiIndexSet(std::vector<MultiIndex>& inds) :
  indices(std::move(inds))
{
  assert(indices.size()>0);
  // make sure all of the multi-indices have the same dimensions
  for( auto it=indices.begin()+1; it!=indices.end(); ++it ) { assert((it-1)->Dimension()==it->Dimension()); }

  // compute the maximum index for each spatial dimension
  maxIndices.resize(indices[0].Dimension());
  std::fill(maxIndices.begin(), maxIndices.end(), 0);
  for( auto it=indices.begin(); it!=indices.end(); ++it ) { 
    for( std::size_t a=0; a<it->Dimension(); ++a ) { maxIndices[a] = std::max(it->alpha[a], maxIndices[a]); }
  }
}

std::unique_ptr<MultiIndexSet> MultiIndexSet::CreateTotalOrder(std::shared_ptr<Parameters> const& para) { return CreateTotalOrder(para->Get<std::size_t>("InputDimension"), para->Get<std::size_t>("MaximumOrder")); }

std::unique_ptr<MultiIndexSet> MultiIndexSet::CreateTotalOrder(std::size_t const dim, std::size_t const maxOrder) {
  // an empty vectoor of indices
  std::vector<MultiIndex> indices;
  
  // start with the zero index 
  std::vector<std::size_t> base(dim, 0);

  // recursively add the total order indices
  CreateTotalOrder(maxOrder, 0, base, indices);

  // create the set
  return std::make_unique<MultiIndexSet>(indices);
}

void MultiIndexSet::CreateTotalOrder(std::size_t const maxOrder, std::size_t const currDim, std::vector<std::size_t>& base, std::vector<MultiIndex>& indices) {
  const std::size_t dim = base.size();
  assert(currDim<dim);

  // the order of the first currDim indices
  std::size_t currOrder = 0;
  for( auto it=base.begin(); it!=base.begin()+currDim+1; ++it ) { currOrder += *it; }
  assert(currOrder<=maxOrder);

  // if we are at the last dimension
  if( currDim==dim-1 ) { 
    // add all of the indices up to the max order
    for( std::size_t i=0; i<=maxOrder-currOrder; ++i ) {
      base[dim-1] = i;
      indices.emplace_back(base);
    }

    return;
  }

  for( std::size_t i=0; i<=maxOrder-currOrder; ++i ) {
    // set the remaining indices to zero
    for( std::size_t j=1; j<=dim-currDim; ++j ) { base[dim-j] = 0; }

    // increment the order of the current dimension (up to the largest possible)
    base[currDim] = i;
    CreateTotalOrder(maxOrder, currDim+1, base, indices);
  }
}

std::size_t MultiIndexSet::Dimension() const { return indices[0].Dimension(); }

std::size_t MultiIndexSet::NumIndices() const { return indices.size(); }

std::size_t MultiIndexSet::MaxIndex(std::size_t const j) const { return maxIndices[j]; }
