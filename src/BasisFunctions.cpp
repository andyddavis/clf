#include "clf/BasisFunctions.hpp"

namespace pt = boost::property_tree;
using namespace clf;

BasisFunctions::BasisFunctions(std::shared_ptr<muq::Utilities::MultiIndexSet> const& multis, pt::ptree const& pt) : multis(multis) {}

std::shared_ptr<BasisFunctions> BasisFunctions::Construct(std::shared_ptr<muq::Utilities::MultiIndexSet> const& multis, pt::ptree const& pt) {
  // get the name of the basis function
  std::string basisName = pt.get<std::string>("Type");

  // try to find the constructor
  auto iter = GetBasisFunctionsMap()->find(basisName);

  // if not, throw an error
  if( iter==GetBasisFunctionsMap()->end() ) { throw exceptions::BasisFunctionsNameConstuctionException(basisName); }

  // call the constructor
  return iter->second(multis, pt);
}

std::shared_ptr<BasisFunctions::BasisFunctionsMap> BasisFunctions::GetBasisFunctionsMap() {
  // define a static map from type to constructor
  static std::shared_ptr<BasisFunctionsMap> map;

  // create the map if the map has not yet been created ...
  if( !map ) {  map = std::make_shared<BasisFunctionsMap>(); }

  return map;
}

std::size_t BasisFunctions::NumBasisFunctions() const { return multis->Size(); }

Eigen::VectorXd BasisFunctions::EvaluateBasisFunctions(Eigen::VectorXd const& x) const {
  Eigen::VectorXd phi(multis->Size());
  for( std::size_t i=0; i<phi.size(); ++i ) { phi(i) = EvaluateBasisFunction(i, x); }
  return phi;
}

double BasisFunctions::EvaluateBasisFunction(std::size_t const ind, Eigen::VectorXd const& x) const {
  assert(ind<multis->Size());

  // get the multi-index
  const Eigen::RowVectorXi iota = multis->at(ind)->GetVector();
  assert(x.size()==iota.size());

  // evaluate the product of basis functions
  double basisEval = 1.0;
  for( std::size_t i=0; i<x.size(); ++i ) { basisEval *= ScalarBasisFunction(iota(i), x(i)); }
  return basisEval;
}
