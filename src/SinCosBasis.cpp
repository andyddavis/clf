#include "clf/SinCosBasis.hpp"

#include <MUQ/Utilities/MultiIndices/MultiIndexFactory.h>

namespace pt = boost::property_tree;
using namespace muq::Utilities;
using namespace clf;

CLF_REGISTER_BASIS_FUNCTION(SinCosBasis)

SinCosBasis::SinCosBasis(std::shared_ptr<MultiIndexSet> const& multis, pt::ptree const& pt) : BasisFunctions(multis, pt) {}

double SinCosBasis::ScalarBasisFunction(std::size_t const ind, double const x) const {
  // constant basis
  if( ind==0 ) { return 1.0; }

  // cosine basis (even numbers)
  if( ind%2==0 ) { return std::cos(M_PI*(ind/2)*x); }

  // sine basis (odd numbers)
  return std::sin(M_PI*((ind+1)/2)*x);
}

std::shared_ptr<SinCosBasis> SinCosBasis::TotalOrderBasis(boost::property_tree::ptree const& pt) {
  std::shared_ptr<MultiIndexSet> multis = MultiIndexFactory::CreateTotalOrder(pt.get<std::size_t>("InputDimension"), 2*pt.get<std::size_t>("Order", 1));

  return std::make_shared<SinCosBasis>(multis, pt);
}
