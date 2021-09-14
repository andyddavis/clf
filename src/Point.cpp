#include "clf/Point.hpp"

using namespace clf;

Point::Point(Eigen::VectorXd const& x) :
x(x)
{}

Point::Point(Eigen::VectorXd const& x, std::shared_ptr<const Model> const& model) :
x(x),
model(model)
{}

Eigen::VectorXd Point::RightHandSide() const {
  assert(model);
  return model->RightHandSide(x);
}

Eigen::VectorXd Point::RightHandSide(Eigen::VectorXd const& loc) const {
  assert(model);
  return model->RightHandSide(loc);
}
