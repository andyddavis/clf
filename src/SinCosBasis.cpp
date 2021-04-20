#include "clf/SinCosBasis.hpp"

#include <MUQ/Utilities/MultiIndices/MultiIndexFactory.h>

namespace pt = boost::property_tree;
using namespace muq::Utilities;
using namespace clf;

CLF_REGISTER_BASIS_FUNCTION(SinCosBasis)

SinCosBasis::SinCosBasis(std::shared_ptr<MultiIndexSet> const& multis, pt::ptree const& pt) : BasisFunctions(multis, pt) {}

std::shared_ptr<SinCosBasis> SinCosBasis::TotalOrderBasis(boost::property_tree::ptree const& pt) {
  std::shared_ptr<MultiIndexSet> multis = MultiIndexFactory::CreateTotalOrder(pt.get<std::size_t>("InputDimension"), 2*pt.get<std::size_t>("Order", 1));

  return std::make_shared<SinCosBasis>(multis, pt);
}

double SinCosBasis::ScalarBasisFunction(double const x, std::size_t const ind) const {
  // constant basis
  if( ind==0 ) { return 1.0; }

  // cosine basis (even numbers)
  if( ind%2==0 ) { return std::cos(M_PI*ind/2*x); }

  // sine basis (odd numbers)
  return std::sin(M_PI*(ind/2+1)*x);
}

double SinCosBasis::ScalarBasisFunctionDerivative(double const x, std::size_t const ind, std::size_t const k) const {
  // constant basis
  if( ind==0 ) { return 0.0; }

  // cosine basis (even numbers)
  if( ind%2==0 ) {
    const double scalar = M_PI*ind/2;
    if( k%4==1 ) {
      return -std::pow(scalar, (double)k)*std::sin(scalar*x);
    } else if( k%4==2 ) {
      return -std::pow(scalar, (double)k)*std::cos(scalar*x);
    } else if( k%4==3 ) {
      return std::pow(scalar, (double)k)*std::sin(scalar*x);
    } else {
      return std::pow(scalar, (double)k)*std::cos(scalar*x);
    }
  }

  const double scalar = M_PI*(ind/2+1);

  // sine basis (odd numbers)
  if( k%4==1 ) {
      return std::pow(scalar, (double)k)*std::cos(scalar*x);
    } else if( k%4==2 ) {
      return -std::pow(scalar, (double)k)*std::sin(scalar*x);
    } else if( k%4==3 ) {
      return -std::pow(scalar, (double)k)*std::cos(scalar*x);
    } else {
      return std::pow(scalar, (double)k)*std::sin(scalar*x);
    }
}
