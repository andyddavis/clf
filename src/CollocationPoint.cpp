#include "clf/CollocationPoint.hpp"

using namespace clf;

CollocationPoint::CollocationPoint(double const weight, Eigen::VectorXd const& x, std::shared_ptr<const Model> const& model) :
Point(x, model),
weight(weight)
{}

Eigen::VectorXd CollocationPoint::Operator() const {
  assert(model);

  // get the support point
  auto pnt = supportPoint.lock();
  assert(pnt);

  return model->Operator(x, pnt->Coefficients(), pnt->GetBasisFunctions());
}

Eigen::VectorXd CollocationPoint::Operator(Eigen::VectorXd const& loc) const {
  assert(model);
  assert(loc.size()==model->inputDimension);

  // get the support point
  auto pnt = supportPoint.lock();
  assert(pnt);

  return model->Operator(loc, pnt->Coefficients(), pnt->GetBasisFunctions());
}

Eigen::VectorXd CollocationPoint::Operator(Eigen::VectorXd const& loc, Eigen::VectorXd const& coeffs) const {
  assert(model);
  assert(loc.size()==model->inputDimension);

  // get the support point
  auto pnt = supportPoint.lock();
  assert(pnt);
  assert(pnt->NumCoefficients()==coeffs.size());

  return model->Operator(loc, coeffs, pnt->GetBasisFunctions());
}

Eigen::MatrixXd CollocationPoint::OperatorJacobian() const {
  assert(model);

  // get the support point
  auto pnt = supportPoint.lock();
  assert(pnt);
  return model->OperatorJacobian(x, pnt->Coefficients(), pnt->GetBasisFunctions());
}

Eigen::MatrixXd CollocationPoint::OperatorJacobian(Eigen::VectorXd const& loc) const {
  assert(model);
  assert(loc.size()==model->inputDimension);

  // get the support point
  auto pnt = supportPoint.lock();
  assert(pnt);

  return model->OperatorJacobian(loc, pnt->Coefficients(), pnt->GetBasisFunctions());
}

Eigen::MatrixXd CollocationPoint::OperatorJacobian(Eigen::VectorXd const& loc, Eigen::VectorXd const& coeffs) const {
  assert(model);
  assert(loc.size()==model->inputDimension);

  // get the support point
  auto pnt = supportPoint.lock();
  assert(pnt);
  assert(pnt->NumCoefficients()==coeffs.size());

  return model->OperatorJacobian(loc, coeffs, pnt->GetBasisFunctions());
}

Eigen::MatrixXd CollocationPoint::OperatorJacobianByFD(Eigen::VectorXd const& loc, Eigen::VectorXd const& coeffs, Model::FDOrder const order, double const fdEps) const {
  assert(model);
  assert(loc.size()==model->inputDimension);

  // get the support point
  auto pnt = supportPoint.lock();
  assert(pnt);
  assert(pnt->NumCoefficients()==coeffs.size());

  return model->OperatorJacobianByFD(loc, coeffs, pnt->GetBasisFunctions(), order, Eigen::VectorXd(), fdEps);
}

std::vector<Eigen::MatrixXd> CollocationPoint::OperatorHessian() const {
  assert(model);

  // get the support point
  auto pnt = supportPoint.lock();
  assert(pnt);
  return model->OperatorHessian(x, pnt->Coefficients(), pnt->GetBasisFunctions());
}

std::vector<Eigen::MatrixXd> CollocationPoint::OperatorHessian(Eigen::VectorXd const& loc) const {
  assert(model);
  assert(loc.size()==model->inputDimension);

  // get the support point
  auto pnt = supportPoint.lock();
  assert(pnt);

  return model->OperatorJacobian(loc, pnt->Coefficients(), pnt->GetBasisFunctions());
}

std::vector<Eigen::MatrixXd> CollocationPoint::OperatorHessian(Eigen::VectorXd const& loc, Eigen::VectorXd const& coefficients) const {
  assert(model);
  assert(loc.size()==model->inputDimension);

  // get the support point
  auto pnt = supportPoint.lock();
  assert(pnt);
  assert(pnt->NumCoefficients()==coeffs.size());

  return model->OperatorJacobian(loc, coeffs, pnt->GetBasisFunctions());
}

std::size_t CollocationPoint::LocalIndex() const { return localIndex; }
