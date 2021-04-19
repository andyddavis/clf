#include "clf/Model.hpp"

namespace pt = boost::property_tree;
using namespace clf;

Model::Model(pt::ptree const& pt) :
inputDimension(pt.get<std::size_t>("InputDimension", 1)),
outputDimension(pt.get<std::size_t>("OutputDimension", 1))
{}

Eigen::VectorXd Model::RightHandSide(Eigen::VectorXd const& x) const {
  assert(x.size()==inputDimension);
  Eigen::VectorXd rhs;

  // try to call the vector implementation
  try {
    rhs = RightHandSideImpl(x);
  } catch( exceptions::ModelHasNotImplementedRHS const& excvector ) {
    try {
      rhs.resize(outputDimension);
      for( std::size_t i=0; i<outputDimension; ++i ) { rhs(i) = RightHandSideImpl(x, i); }
    } catch( exceptions::ModelHasNotImplementedRHS const& exccomponent ) {
      throw exceptions::ModelHasNotImplementedRHS(exceptions::ModelHasNotImplementedRHS::Type::BOTH);
    }
  }

  return rhs;
}

Eigen::VectorXd Model::RightHandSideImpl(Eigen::VectorXd const& x) const {
  throw exceptions::ModelHasNotImplementedRHS(exceptions::ModelHasNotImplementedRHS::Type::VECTOR);
  return Eigen::VectorXd();
}

double Model::RightHandSideImpl(Eigen::VectorXd const& x, std::size_t const outind) const {
  throw exceptions::ModelHasNotImplementedRHS(exceptions::ModelHasNotImplementedRHS::Type::COMPONENT);
  return 0.0;
}
