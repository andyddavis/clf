#include "clf/PolynomialBasis.hpp"

namespace pt = boost::property_tree;
using namespace muq::Utilities;
using namespace muq::Approximation;
using namespace clf;

CLF_REGISTER_BASIS_FUNCTION(PolynomialBasis)

PolynomialBasis::PolynomialBasis(std::shared_ptr<MultiIndexSet> const& multis, pt::ptree const& pt) : BasisFunctions(multis, pt) {
  poly = IndexedScalarBasis::Construct(pt.get<std::string>("ScalarBasis", "Legendre"));
}

double PolynomialBasis::ScalarBasisFunction(std::size_t const ind, double const x) const { return poly->BasisEvaluate(ind, x); }
