#include "clf/BasisFunctions.hpp"

namespace pt = boost::property_tree;
using namespace clf;

BasisFunctions::BasisFunctions(pt::ptree const& pt) {}

std::shared_ptr<BasisFunctions> BasisFunctions::Construct(pt::ptree const& pt) {
  // get the name of the basis function
  std::string basisName = pt.get<std::string>("Type");

  // try to find the constructor
  auto iter = GetBasisFunctionsMap()->find(basisName);

  // if not, throw an error
  if( iter==GetBasisFunctionsMap()->end() ) { throw BasisFunctionsNameConstuctionException(basisName); }

  // call the constructor
  return iter->second(pt);
}

std::shared_ptr<BasisFunctions::BasisFunctionsMap> BasisFunctions::GetBasisFunctionsMap() {
  // define a static map from type to constructor
  static std::shared_ptr<BasisFunctionsMap> map;

  // create the map if the map has not yet been created ...
  if( !map ) {  map = std::make_shared<BasisFunctionsMap>(); }

  return map;
}
