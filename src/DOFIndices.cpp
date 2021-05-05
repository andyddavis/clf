#include "clf/DOFIndices.hpp"

using namespace clf;

DOFIndices::DOFIndices(std::shared_ptr<SupportPointCloud> const& cloud) :
cloud(cloud),
globalDoFIndices(GlobalDOFIndices(cloud))
{}

std::vector<std::size_t> DOFIndices::GlobalDOFIndices(std::shared_ptr<SupportPointCloud> const& cloud) {
  // loop through the support points
  std::vector<std::size_t> globalDoFIndices(cloud->NumSupportPoints());
  std::size_t ind = 0;
  for( auto point=cloud->Begin(); point!=cloud->End(); ++point ) {
    assert((*point)->GlobalIndex()<globalDoFIndices.size());
    globalDoFIndices[(*point)->GlobalIndex()] = ind;
    ind += (*point)->NumCoefficients();
  }

  return globalDoFIndices;
}
