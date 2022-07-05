#include "clf/LinearSystem.hpp"

using namespace clf;

LinearSystem::LinearSystem(std::size_t const indim, std::size_t const outdim) : 
  SystemOfEquations(indim, outdim), 
  matdim(outdim)
{}

LinearSystem::LinearSystem(std::size_t const indim, std::size_t const outdim, std::size_t const matdim) :
  SystemOfEquations(indim, outdim), 
  matdim(matdim)
{}

LinearSystem::LinearSystem(std::size_t const indim, Eigen::MatrixXd const A) :
  SystemOfEquations(indim, A.rows()), 
  matdim(A.cols()),
  A(A)
{}

Eigen::MatrixXd LinearSystem::Operator(Eigen::VectorXd const& x) const { 
  if( A ) { return *A; }
  return Eigen::MatrixXd::Identity(outdim, matdim); 
}

Eigen::VectorXd LinearSystem::Operator(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x) const {
  assert(u->InputDimension()==indim);
  assert(u->OutputDimension()==matdim);
  assert(x.size()==indim);
  return Operator(x)*u->Evaluate(x);
}
