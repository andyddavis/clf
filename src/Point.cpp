#include "clf/Point.hpp"

using namespace clf;

std::atomic<std::size_t> Point::nextID = 0;

Point::Point(Eigen::VectorXd const& x) :
  x(x), id(nextID++)
{}

Point::Point(Eigen::VectorXd const& x, Eigen::VectorXd const& normal) :
  x(x), normal(normal), id(nextID++)
{}

Point::Point(std::pair<Eigen::VectorXd, Eigen::VectorXd> const& x) :
  x(x.first), normal(x.second), id(nextID++)
{}
