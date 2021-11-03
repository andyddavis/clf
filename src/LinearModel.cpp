#include "clf/LinearModel.hpp"

namespace pt = boost::property_tree;
using namespace clf;

LinearModel::LinearModel(std::size_t const indim, std::size_t const outdim) :
Model(indim, outdim)
{}

LinearModel::LinearModel(pt::ptree const& pt) : 
Model(pt)
{}

Eigen::VectorXd LinearModel::OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
  try { // try to use matrix implementation 
    return ModelMatrix(x, bases)*coefficients;
  } catch( exceptions::ModelHasNotImplemented const& exc ) {
    return IdentityOperator(x, coefficients, bases);
  }
}

Eigen::MatrixXd LinearModel::OperatorJacobianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
  try { // try to use matrix implementation 
    return ModelMatrix(x, bases);
  } catch( exceptions::ModelHasNotImplemented const& exc ) {
    return IdentityOperatorJacobian(x, bases);
  }
}

std::vector<Eigen::MatrixXd> LinearModel::OperatorHessianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
  std::vector<Eigen::MatrixXd> hess(outputDimension);
  for( auto& it : hess ) { it = Eigen::MatrixXd::Zero(coefficients.size(), coefficients.size()); }
  return hess;
}

Eigen::MatrixXd LinearModel::ModelMatrix(Eigen::VectorXd const& x, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const {
  throw exceptions::ModelHasNotImplemented(exceptions::ModelHasNotImplemented::Type::BOTH, exceptions::ModelHasNotImplemented::Function::LINEAR_MODEL_MATRIX);
  return Eigen::MatrixXd();
}

bool LinearModel::IsLinear() const { return true; }
