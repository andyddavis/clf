#include "clf/PolynomialBasis.hpp"

#include <MUQ/Utilities/MultiIndices/MultiIndexFactory.h>

namespace pt = boost::property_tree;
using namespace muq::Utilities;
using namespace muq::Approximation;
using namespace clf;

CLF_REGISTER_BASIS_FUNCTION(PolynomialBasis)

PolynomialBasis::PolynomialBasis(std::shared_ptr<MultiIndexSet> const& multis, pt::ptree const& pt) : BasisFunctions(multis, pt) {
  poly = IndexedScalarBasis::Construct(pt.get<std::string>("ScalarBasis", "ProbabilistHermite"));
}

std::shared_ptr<PolynomialBasis> PolynomialBasis::TotalOrderBasis(pt::ptree const& pt) {
  std::shared_ptr<MultiIndexSet> multis = MultiIndexFactory::CreateTotalOrder(pt.get<std::size_t>("InputDimension"), pt.get<std::size_t>("Order", 2));

  return std::make_shared<PolynomialBasis>(multis, pt);
}

double PolynomialBasis::ScalarBasisFunction(double const x, std::size_t const ind, std::size_t const coordinate) const {
  assert(poly);
  return poly->BasisEvaluate(ind, x);
}

double PolynomialBasis::ScalarBasisFunctionDerivative(double const x, std::size_t const ind, std::size_t const coordinate, std::size_t const k) const {
  assert(poly);
  if( k==0 ) { return ScalarBasisFunction(x, ind, coordinate); }
  return poly->DerivativeEvaluate(ind, k, x);
}
