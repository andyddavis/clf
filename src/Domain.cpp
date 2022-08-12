#include "clf/Domain.hpp"

#include "clf/CLFExceptions.hpp"

using namespace clf;

Domain::Domain(std::size_t const dim) :
  dim(dim)
{}

bool Domain::Inside(Eigen::VectorXd const& x) const {
   throw exceptions::NotImplemented("Domain::Inside");
   return false;
}

Eigen::VectorXd Domain::MapToHypercube(Eigen::VectorXd const& x) const {
  throw exceptions::NotImplemented("Domain::MapToHypercube");
  return Eigen::VectorXd();
}

Eigen::VectorXd Domain::Sample() {
  throw exceptions::NotImplemented("Domain::Sample");
  return Eigen::VectorXd();
}

double Domain::Distance(Eigen::VectorXd const& x1, Eigen::VectorXd const& x2) const { return (x1-x2).norm(); }
