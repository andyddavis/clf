#include "clf/PenaltyFunction.hpp"

using namespace clf;

DensePenaltyFunction::DensePenaltyFunction(std::size_t const indim, std::size_t const outdim, std::shared_ptr<Parameters> const& para) :
  PenaltyFunction<Eigen::MatrixXd>(indim, outdim, para)
{}

Eigen::MatrixXd DensePenaltyFunction::JacobianFD(Eigen::VectorXd beta) {
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(outdim, indim);
  
  const double delta = para->Get<double>("DeltaJacobian", 1.0e-2);
  const Eigen::VectorXd weights = FiniteDifferenceWeights(para->Get<std::size_t>("OrderJacobian", 8));
  
  for( std::size_t i=0; i<indim; ++i ) {
    for( std::size_t j=0; j<weights.size(); ++j ) {
      beta(i) += delta;
      jac.col(i) += weights(j)*Evaluate(beta);
    }
    beta(i) -= weights.size()*delta;
    for( std::size_t j=0; j<weights.size(); ++j ) {
      beta(i) -= delta;
      jac.col(i) -= weights(j)*Evaluate(beta);
    }
    beta(i) += weights.size()*delta;
  }
  jac /= delta;
  
  return jac;
}

SparsePenaltyFunction::SparsePenaltyFunction(std::size_t const indim, std::size_t const outdim, std::shared_ptr<Parameters> const& para) :
  PenaltyFunction<Eigen::SparseMatrix<double> >(indim, outdim, para)
{}

Eigen::SparseMatrix<double> SparsePenaltyFunction::JacobianFD(Eigen::VectorXd beta) {
  const double delta = para->Get<double>("DeltaJacobian", 1.0e-2);
  const Eigen::VectorXd weights = FiniteDifferenceWeights(para->Get<std::size_t>("OrderJacobian", 8));
  
  std::vector<Eigen::Triplet<double> > entries;
  for( std::size_t i=0; i<indim; ++i ) {
    Eigen::VectorXd vec = Eigen::VectorXd::Zero(outdim);
    
    for( std::size_t j=0; j<weights.size(); ++j ) {
      beta(i) += delta;
      vec += weights(j)*Evaluate(beta);
    }
    beta(i) -= weights.size()*delta;
    for( std::size_t j=0; j<weights.size(); ++j ) {
      beta(i) -= delta;
      vec -= weights(j)*Evaluate(beta);
    }
    beta(i) += weights.size()*delta;
    
    for( std::size_t j=0; j<outdim; ++j ) {
      if( std::abs(vec(j))>1.0e-14 ) { entries.emplace_back(j, i, vec(j)/delta); }
    }
  }
  
  Eigen::SparseMatrix<double> jac(outdim, indim);
  jac.setFromTriplets(entries.begin(), entries.end());
  return jac;
}
