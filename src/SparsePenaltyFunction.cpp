#include "clf/SparsePenaltyFunction.hpp"

using namespace clf;

SparsePenaltyFunction::SparsePenaltyFunction(std::size_t const indim, std::size_t const outdim, std::shared_ptr<const Parameters> const& para) :
  PenaltyFunction<Eigen::SparseMatrix<double> >(indim, outdim, para)
{}

Eigen::SparseMatrix<double> SparsePenaltyFunction::Jacobian(Eigen::VectorXd const& beta) {
  const std::vector<Eigen::Triplet<double> > entries = JacobianEntries(beta);
  
  Eigen::SparseMatrix<double> jac(outdim, indim);
  jac.setFromTriplets(entries.begin(), entries.end());
  return jac;
}

Eigen::SparseMatrix<double> SparsePenaltyFunction::JacobianFD(Eigen::VectorXd const& beta) {
  const std::vector<Eigen::Triplet<double> > entries = JacobianEntriesFD(beta);
  
  Eigen::SparseMatrix<double> jac(outdim, indim);
  jac.setFromTriplets(entries.begin(), entries.end());
  return jac;
}

std::vector<Eigen::Triplet<double> > SparsePenaltyFunction::JacobianEntries(Eigen::VectorXd const& beta) { return JacobianEntriesFD(beta); }

std::vector<Eigen::Triplet<double> > SparsePenaltyFunction::JacobianEntriesFD(Eigen::VectorXd const& beta) {
  const double delta = para->Get<double>("DeltaFD", deltaFD_DEFAULT);
  const Eigen::VectorXd weights = FiniteDifference::Weights(para->Get<std::size_t>("OrderFD", orderFD_DEFAULT));
  
  std::vector<Eigen::Triplet<double> > entries;
  Eigen::VectorXd b = beta;
  const double sparseTol = para->Get<double>("SparsityTolerance", sparsityTolerance_DEFAULT);
  for( std::size_t i=0; i<indim; ++i ) {
    const Eigen::VectorXd vec = FirstDerivativeFD(i, delta, weights, b);
        
    for( std::size_t j=0; j<outdim; ++j ) {
      if( std::abs(vec(j))>sparseTol ) { entries.emplace_back(j, i, vec(j)); }
    }
  }

  return entries;
}

std::vector<Eigen::Triplet<double> > SparsePenaltyFunction::HessianEntries(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) { return HessianEntriesFD(beta, weights); }

std::vector<Eigen::Triplet<double> > SparsePenaltyFunction::HessianEntriesFD(Eigen::VectorXd const& beta, Eigen::VectorXd const& sumWeights) {
  assert(sumWeights.size()==outdim);
  const double delta = para->Get<double>("DeltaFD", deltaFD_DEFAULT);
  const Eigen::VectorXd weights = FiniteDifference::Weights(para->Get<std::size_t>("OrderFD", orderFD_DEFAULT));

  Eigen::VectorXd b = beta;
  const double sparseTol = para->Get<double>("SparsityTolerance", sparsityTolerance_DEFAULT);
  std::vector<Eigen::Triplet<double> > entries;
  for( std::size_t i=0; i<indim; ++i ) {
    for( std::size_t j=0; j<weights.size(); ++j ) {
      b(i) += delta;
      const std::vector<Eigen::Triplet<double> > local = JacobianEntries(b);
      for( const auto& it : local ) {
	if( std::abs(it.value())>sparseTol ) { entries.emplace_back(i, it.col(), sumWeights(it.row())*weights(j)*it.value()/delta); }
      }
    }
    b(i) -= weights.size()*delta;
    for( std::size_t j=0; j<weights.size(); ++j ) {
      b(i) -= delta;
      const std::vector<Eigen::Triplet<double> > local = JacobianEntries(b);
      for( const auto& it : local ) {
	if( std::abs(it.value())>sparseTol ) { entries.emplace_back(i, it.col(), -sumWeights(it.row())*weights(j)*it.value()/delta); }
      }
    }
    b(i) += weights.size()*delta;
  }

  return entries;
}

Eigen::SparseMatrix<double> SparsePenaltyFunction::Hessian(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) {
  assert(weights.size()==outdim);
  const std::vector<Eigen::Triplet<double> > entries = HessianEntries(beta, weights);

  Eigen::SparseMatrix<double> hess(indim, indim);
  hess.setFromTriplets(entries.begin(), entries.end());
  return hess;
}

Eigen::SparseMatrix<double> SparsePenaltyFunction::HessianFD(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) {
  const std::vector<Eigen::Triplet<double> > entries = HessianEntriesFD(beta, weights);

  Eigen::SparseMatrix<double> hess(indim, indim);
  hess.setFromTriplets(entries.begin(), entries.end());
  return hess;
}
