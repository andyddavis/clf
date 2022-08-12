#include "clf/Point.hpp"

using namespace clf;

std::atomic<std::size_t> Point::nextID = 0;

Point::Point(Eigen::VectorXd const& x) :
  x(x), id(nextID++)
{}
