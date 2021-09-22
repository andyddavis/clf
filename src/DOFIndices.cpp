#include "clf/DOFIndices.hpp"

using namespace clf;

DOFIndices::DOFIndices(std::shared_ptr<SupportPointCloud> const& cloud) :
cloud(cloud),
globalDoFIndices(GlobalDOFIndices(cloud)),
maxNonZeros(MaxNonZeros(cloud))
{}

std::vector<std::size_t> DOFIndices::GlobalDOFIndices(std::shared_ptr<SupportPointCloud> const& cloud) {
  // loop through the support points
  std::vector<std::size_t> globalDoFIndices(cloud->NumPoints());
  std::size_t ind = 0;
  for( auto it=cloud->Begin(); it!=cloud->End(); ++it ) {
    auto point = std::dynamic_pointer_cast<SupportPoint>(*it);
    assert(point);

    assert(point->GlobalIndex()<globalDoFIndices.size());
    globalDoFIndices[point->GlobalIndex()] = ind;
    ind += point->NumCoefficients();
  }

  return globalDoFIndices;
}

std::size_t DOFIndices::MaxNonZeros(std::shared_ptr<SupportPointCloud> const& cloud) {
  std::size_t nonZeros = 0;
  for( auto it=cloud->Begin(); it!=cloud->End(); ++it ) {
    auto point = std::dynamic_pointer_cast<SupportPoint>(*it);
    assert(point);

    // if the uncoupled hessian where dense
    nonZeros += point->NumCoefficients()*point->NumCoefficients();

    /*if( (*point)->couplingScale>0.0 ) {
      for( const auto& neighInd : (*point)->GlobalNeighborIndices() ) {
        const auto& neigh = cloud->GetSupportPoint(neighInd);
        assert(neigh);
        const std::vector<std::shared_ptr<const BasisFunctions> >& pointBases = (*point)->GetBasisFunctions();
        const std::vector<std::shared_ptr<const BasisFunctions> >& neighBases = neigh->GetBasisFunctions();
        assert(pointBases.size()==neighBases.size());
        for( std::size_t i=0; i<pointBases.size(); ++i ) { nonZeros += 2*pointBases[i]->NumBasisFunctions()*neighBases[i]->NumBasisFunctions(); }
      }
    }*/
  }

  return nonZeros;
}
