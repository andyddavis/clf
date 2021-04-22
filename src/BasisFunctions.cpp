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

double BasisFunctions::FunctionEvaluation(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients) const {
  assert(coefficients.size()==NumBasisFunctions());
  return coefficients.dot(EvaluateBasisFunctions(x));
}

Eigen::VectorXd BasisFunctions::EvaluateBasisFunctions(Eigen::VectorXd const& x) const {
  Eigen::VectorXd phi(multis->Size());
  for( std::size_t i=0; i<phi.size(); ++i ) { phi(i) = EvaluateBasisFunction(x, i); }
  return phi;
}

double BasisFunctions::EvaluateBasisFunction(Eigen::VectorXd const& x, std::size_t const ind) const {
  assert(ind<multis->Size());

  // get the multi-index
  const Eigen::RowVectorXi iota = multis->at(ind)->GetVector();
  assert(x.size()==iota.size());

  // evaluate the product of basis functions
  double basisEval = 1.0;
  for( std::size_t i=0; i<x.size(); ++i ) {
    if( std::abs(basisEval)<1.0e-14 ) { return 0.0; }
    basisEval *= ScalarBasisFunction(x(i), iota(i), i);
  }
  return basisEval;
}

Eigen::MatrixXd BasisFunctions::EvaluateBasisFunctionDerivatives(Eigen::VectorXd const& x, std::size_t const k) const {
  Eigen::MatrixXd phi(multis->GetMultiLength(), multis->Size());
  for( std::size_t i=0; i<multis->GetMultiLength(); ++i ) { phi.row(i) = EvaluateBasisFunctionDerivatives(x, i, k); }
  return phi;
}

Eigen::VectorXd BasisFunctions::EvaluateBasisFunctionDerivatives(Eigen::VectorXd const& x, std::size_t const p, std::size_t const k) const {
  Eigen::VectorXd phi(multis->Size());
  for( std::size_t i=0; i<phi.size(); ++i ) { phi(i) = EvaluateBasisFunctionDerivative(x, i, p, k); }
  return phi;
}

double BasisFunctions::EvaluateBasisFunctionDerivative(Eigen::VectorXd const& x, std::size_t const ind, std::size_t const p, std::size_t const k) const {
  assert(multis);
  assert(ind<multis->Size());
  assert(p<x.size());

  // get the multi-index
  assert(multis->at(ind));
  const Eigen::RowVectorXi iota = multis->at(ind)->GetVector();
  assert(x.size()==iota.size());

  // evaluate the product of basis functions
  double basisEval = 1.0;
  for( std::size_t i=0; i<x.size(); ++i ) {
    if( std::abs(basisEval)<1.0e-14 ) { return 0.0; }
    if( i==p ) {
      basisEval *= ScalarBasisFunctionDerivative(x(i), iota(i), i, k);
    } else {
      basisEval *= ScalarBasisFunction(x(i), iota(i), i);
    }
  }
  return basisEval;
}
