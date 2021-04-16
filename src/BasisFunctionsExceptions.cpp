#include "clf/BasisFunctionsExceptions.hpp"

#include "clf/BasisFunctions.hpp"

using namespace clf::exceptions;

BasisFunctionsNameConstuctionException::BasisFunctionsNameConstuctionException(std::string const& basisName) : CLFException(), basisName(basisName) {
  message = "ERROR: Could not find basis functions \"" + basisName + "\".  Available options are:\n";
  auto map = BasisFunctions::GetBasisFunctionsMap();
  for( auto it=map->begin(); it!=map->end(); ++it ) { message += it->first + "\n"; }
  message += "(clf::exceptions::BasisFunctionsNameConstuctionException).";
}
