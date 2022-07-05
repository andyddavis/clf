#include "clf/SystemOfEquations.hpp"

using namespace clf;

SystemOfEquations::SystemOfEquations(std::size_t const indim, std::size_t const outdim) :
  indim(indim), outdim(outdim)  
{}

Eigen::VectorXd SystemOfEquations::RightHandSide(Eigen::VectorXd const& x) const { return Eigen::VectorXd::Zero(outdim); }

Eigen::VectorXd SystemOfEquations::Operator(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x) const { 
  assert(u->InputDimension()==indim);
  assert(u->OutputDimension()==outdim);
  assert(x.size()==indim);
  return u->Evaluate(x);
}
