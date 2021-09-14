#include "clf/ColocationPoint.hpp"

using namespace clf;

ColocationPoint::ColocationPoint(Eigen::VectorXd const& x, std::shared_ptr<const Model> const& model) :
Point(x, model)
{}

Eigen::VectorXd ColocationPoint::Operator() const {
  assert(model);

  // get the support point
  auto pnt = supportPoint.lock();
  assert(pnt);

  return model->Operator(x, pnt->Coefficients(), pnt->GetBasisFunctions());
}

Eigen::VectorXd ColocationPoint::Operator(Eigen::VectorXd const& loc) const {
  assert(model);
  assert(loc.size()==model->inputDimension);

  // get the support point
  auto pnt = supportPoint.lock();
  assert(pnt);

  return model->Operator(loc, pnt->Coefficients(), pnt->GetBasisFunctions());
}

Eigen::VectorXd ColocationPoint::Operator(Eigen::VectorXd const& loc, Eigen::VectorXd const& coeffs) const {
  assert(model);
  assert(loc.size()==model->inputDimension);

  // get the support point
  auto pnt = supportPoint.lock();
  assert(pnt);
  assert(pnt->NumCoefficients()==coeffs.size());

  return model->Operator(loc, coeffs, pnt->GetBasisFunctions());
}

Eigen::MatrixXd ColocationPoint::OperatorJacobian() const {
  assert(model);

  // get the support point
  auto pnt = supportPoint.lock();
  assert(pnt);
  return model->OperatorJacobian(x, pnt->Coefficients(), pnt->GetBasisFunctions());
}

Eigen::MatrixXd ColocationPoint::OperatorJacobian(Eigen::VectorXd const& loc) const {
  assert(model);
  assert(loc.size()==model->inputDimension);

  // get the support point
  auto pnt = supportPoint.lock();
  assert(pnt);

  return model->OperatorJacobian(loc, pnt->Coefficients(), pnt->GetBasisFunctions());
}

Eigen::MatrixXd ColocationPoint::OperatorJacobian(Eigen::VectorXd const& loc, Eigen::VectorXd const& coeffs) const {
  assert(model);
  assert(loc.size()==model->inputDimension);

  // get the support point
  auto pnt = supportPoint.lock();
  assert(pnt);
  assert(pnt->NumCoefficients()==coeffs.size());

  return model->OperatorJacobian(loc, coeffs, pnt->GetBasisFunctions());
}
