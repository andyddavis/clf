#include "clf/PenaltyFunction.hpp"

using namespace clf;

DensePenaltyFunction::DensePenaltyFunction(std::size_t const indim, std::size_t const outdim, std::shared_ptr<Parameters> const& para) :
  PenaltyFunction<Eigen::MatrixXd>(indim, outdim, para)
{}

void DensePenaltyFunction::JacobianFD(Eigen::VectorXd const& beta, Eigen::MatrixXd& jac) {
  if( jac.rows()!=outdim || jac.cols()!=indim ) {
    jac = Eigen::MatrixXd::Zero(outdim, indim);
  } else {
    jac.setZero();
  }
  
  const double delta = para->Get<double>("DeltaJacobian", deltaFD_DEFAULT);
  const Eigen::VectorXd weights = FiniteDifferenceWeights(para->Get<std::size_t>("OrderJacobian", orderFD_DEFAULT));
  
  Eigen::VectorXd b = beta;
  for( std::size_t i=0; i<indim; ++i ) { jac.col(i) = FirstDerivativeFD(i, delta, weights, b); }
  jac /= delta;
}

Eigen::MatrixXd DensePenaltyFunction::HessianFD(Eigen::VectorXd const& beta, std::size_t const component) {
  Eigen::MatrixXd hess(indim, indim);

  return hess;
}

SparsePenaltyFunction::SparsePenaltyFunction(std::size_t const indim, std::size_t const outdim, std::shared_ptr<Parameters> const& para) :
  PenaltyFunction<Eigen::SparseMatrix<double> >(indim, outdim, para)
{}

Eigen::SparseMatrix<double> SparsePenaltyFunction::Jacobian(Eigen::VectorXd const& beta) {
  std::vector<Eigen::Triplet<double> > entries;
  JacobianEntries(beta, entries);
  
  Eigen::SparseMatrix<double> jac(outdim, indim);
  jac.setFromTriplets(entries.begin(), entries.end());
  return jac;
}

void SparsePenaltyFunction::JacobianFD(Eigen::VectorXd const& beta, Eigen::SparseMatrix<double>& jac) {
  std::vector<Eigen::Triplet<double> > entries;
  JacobianEntriesFD(beta, entries);
  
  if( jac.rows()!=outdim || jac.cols()!=indim ) { jac.resize(outdim, indim); }
  jac.setZero();
  jac.setFromTriplets(entries.begin(), entries.end());
}

void SparsePenaltyFunction::JacobianEntries(Eigen::VectorXd const& beta, std::vector<Eigen::Triplet<double> >& entries) { JacobianEntriesFD(beta, entries); }

void SparsePenaltyFunction::JacobianEntriesFD(Eigen::VectorXd const& beta, std::vector<Eigen::Triplet<double> >& entries) {
  const double delta = para->Get<double>("DeltaJacobian", deltaFD_DEFAULT);
  const Eigen::VectorXd weights = FiniteDifferenceWeights(para->Get<std::size_t>("OrderJacobian", orderFD_DEFAULT));
  
  entries.clear();
  Eigen::VectorXd b = beta;
  const double sparseTol = para->Get<double>("SparsityTolerance", sparsityTolerance_DEFAULT);
  for( std::size_t i=0; i<indim; ++i ) {
    const Eigen::VectorXd vec = FirstDerivativeFD(i, delta, weights, b);
        
    for( std::size_t j=0; j<outdim; ++j ) {
      if( std::abs(vec(j))>sparseTol ) { entries.emplace_back(j, i, vec(j)/delta); }
    }
  }
}

Eigen::SparseMatrix<double> SparsePenaltyFunction::HessianFD(Eigen::VectorXd const& beta, std::size_t const component) {
  return Eigen::SparseMatrix<double>();
}
