#include "clf/SinCosBasis.hpp"

namespace pt = boost::property_tree;
using namespace clf;

CLF_REGISTER_BASIS_FUNCTION(SinCosBasis)

SinCosBasis::SinCosBasis(pt::ptree const& pt) : BasisFunctions(pt) {}
