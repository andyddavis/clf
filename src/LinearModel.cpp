#include "clf/LinearModel.hpp"

using namespace clf;

LinearModel::LinearModel(std::size_t const indim, std::size_t const outdim) : 
  SystemOfEquations(indim, outdim), 
  matdim(outdim)
{}

LinearModel::LinearModel(std::size_t const indim, std::size_t const outdim, std::size_t const matdim) :
  SystemOfEquations(indim, outdim), 
  matdim(matdim)
{}

LinearModel::LinearModel(std::size_t const indim, Eigen::MatrixXd const A) :
  SystemOfEquations(indim, A.rows()), 
  matdim(A.cols()),
  A(A)
{}

Eigen::MatrixXd LinearModel::Operator(Eigen::VectorXd const& x) const { 
  if( A ) { return *A; }
  return Eigen::MatrixXd::Identity(outdim, matdim); 
}

Eigen::VectorXd LinearModel::Operator(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const {
  assert(u->InputDimension()==indim);
  assert(u->OutputDimension()==matdim);
  assert(u->NumCoefficients()==coeff.size());
  assert(x.size()==indim);
  return Operator(x)*u->Evaluate(x, coeff);
}

Eigen::MatrixXd LinearModel::JacobianWRTCoefficients(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const {
  assert(u->InputDimension()==indim);
  assert(u->OutputDimension()==matdim);
  assert(u->NumCoefficients()==coeff.size());
  assert(x.size()==indim);

  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(outdim, coeff.size());
  const Eigen::MatrixXd A = Operator(x);

  std::size_t start = 0;
  std::size_t basis = 0;
  for( auto it=u->featureMatrix->Begin(); it!=u->featureMatrix->End(); ++it ) {
    const Eigen::RowVectorXd phi = it->first->Evaluate(x).transpose();

    for( std::size_t i=0; i<it->second; ++i ) {
      jac.block(0, start, outdim, phi.size()) = A.col(basis++)*phi;
      
      start += phi.size();
    }
  }

  return jac;
}

Eigen::MatrixXd LinearModel::HessianWRTCoefficients(std::shared_ptr<LocalFunction> const& u, Eigen::VectorXd const& x, Eigen::VectorXd const& coeff, Eigen::VectorXd const& weights) const { return Eigen::MatrixXd::Zero(coeff.size(), coeff.size()); }
