#include "clf/BasisFunctionsExceptions.hpp"

#include "clf/BasisFunctions.hpp"

using namespace clf;

BasisFunctionsNameConstuctionException::BasisFunctionsNameConstuctionException(std::string const& basisName) : basisName(basisName) {
  message = "ERROR: Could not find basis functions \"" + basisName + "\".  Available options are:\n";
  auto map = BasisFunctions::GetBasisFunctionsMap();
  for( auto it=map->begin(); it!=map->end(); ++it ) { message += it->first + "\n"; }
  message += "(clf::BasisFunctionsNameConstuctionException).";
}

const char* BasisFunctionsNameConstuctionException::what() const noexcept { return message.c_str(); }
