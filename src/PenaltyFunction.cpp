#include "clf/PenaltyFunction.hpp"

using namespace clf;

DensePenaltyFunction::DensePenaltyFunction(std::size_t const indim, std::size_t const outdim, std::shared_ptr<Parameters> const& para) :
  PenaltyFunction<Eigen::MatrixXd>(indim, outdim, para)
{}

Eigen::MatrixXd DensePenaltyFunction::JacobianFD(Eigen::VectorXd const& beta) {
  const double delta = para->Get<double>("DeltaFD", deltaFD_DEFAULT);
  const Eigen::VectorXd weights = FiniteDifferenceWeights(para->Get<std::size_t>("OrderFD", orderFD_DEFAULT));
  
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(outdim, indim);
  Eigen::VectorXd b = beta;
  for( std::size_t i=0; i<indim; ++i ) { jac.col(i) = FirstDerivativeFD(i, delta, weights, b); }
  jac /= delta;
  return jac;
}

Eigen::MatrixXd DensePenaltyFunction::HessianFD(Eigen::VectorXd const& beta, Eigen::VectorXd const& sumWeights) {
  assert(sumWeights.size()==outdim);
  Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(indim, indim);

  const double delta = para->Get<double>("DeltaFD", deltaFD_DEFAULT);
  const Eigen::VectorXd weights = FiniteDifferenceWeights(para->Get<std::size_t>("OrderFD", orderFD_DEFAULT));

  Eigen::VectorXd b = beta;
  for( std::size_t i=0; i<indim; ++i ) {
    Eigen::MatrixXd secondDeriv = Eigen::MatrixXd::Zero(outdim, indim);
    for( std::size_t j=0; j<weights.size(); ++j ) {
      b(i) += delta;
      secondDeriv += weights(j)*Jacobian(b);
    }
    b(i) -= weights.size()*delta;
    for( std::size_t j=0; j<weights.size(); ++j ) {
      b(i) -= delta;
      secondDeriv -= weights(j)*Jacobian(b);
    }
    b(i) += weights.size()*delta;
    secondDeriv /= delta;
    
    for( std::size_t j=0; j<outdim; ++j ) { hess.row(i) += sumWeights(j)*secondDeriv.row(j); }
  }

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

Eigen::SparseMatrix<double> SparsePenaltyFunction::JacobianFD(Eigen::VectorXd const& beta) {
  std::vector<Eigen::Triplet<double> > entries;
  JacobianEntriesFD(beta, entries);
  
  Eigen::SparseMatrix<double> jac(outdim, indim);
  jac.setFromTriplets(entries.begin(), entries.end());
  return jac;
}

void SparsePenaltyFunction::JacobianEntries(Eigen::VectorXd const& beta, std::vector<Eigen::Triplet<double> >& entries) { JacobianEntriesFD(beta, entries); }

void SparsePenaltyFunction::JacobianEntriesFD(Eigen::VectorXd const& beta, std::vector<Eigen::Triplet<double> >& entries) {
  const double delta = para->Get<double>("DeltaFD", deltaFD_DEFAULT);
  const Eigen::VectorXd weights = FiniteDifferenceWeights(para->Get<std::size_t>("OrderFD", orderFD_DEFAULT));
  
  entries.clear();
  Eigen::VectorXd b = beta;
  const double sparseTol = para->Get<double>("SparsityTolerance", sparsityTolerance_DEFAULT);
  entries.clear();
  for( std::size_t i=0; i<indim; ++i ) {
    const Eigen::VectorXd vec = FirstDerivativeFD(i, delta, weights, b);
        
    for( std::size_t j=0; j<outdim; ++j ) {
      if( std::abs(vec(j))>sparseTol ) { entries.emplace_back(j, i, vec(j)/delta); }
    }
  }
}

void SparsePenaltyFunction::HessianEntries(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights, std::vector<Eigen::Triplet<double> >& entries) { HessianEntriesFD(beta, weights, entries); }

void SparsePenaltyFunction::HessianEntriesFD(Eigen::VectorXd const& beta, Eigen::VectorXd const& sumWeights, std::vector<Eigen::Triplet<double> >& entries) {
  assert(sumWeights.size()==outdim);
  const double delta = para->Get<double>("DeltaFD", deltaFD_DEFAULT);
  const Eigen::VectorXd weights = FiniteDifferenceWeights(para->Get<std::size_t>("OrderFD", orderFD_DEFAULT));

  Eigen::VectorXd b = beta;
  const double sparseTol = para->Get<double>("SparsityTolerance", sparsityTolerance_DEFAULT);
  entries.clear();
  for( std::size_t i=0; i<indim; ++i ) {
    for( std::size_t j=0; j<weights.size(); ++j ) {
      b(i) += delta;
      std::vector<Eigen::Triplet<double> > local;
      JacobianEntries(b, local);
      for( const auto& it : local ) {
	if( std::abs(it.value())>sparseTol ) { entries.emplace_back(i, it.col(), sumWeights(it.row())*weights(j)*it.value()/delta); }
      }
    }
    b(i) -= weights.size()*delta;
    for( std::size_t j=0; j<weights.size(); ++j ) {
      b(i) -= delta;
      std::vector<Eigen::Triplet<double> > local;
      JacobianEntries(b, local);
      for( const auto& it : local ) {
	if( std::abs(it.value())>sparseTol ) { entries.emplace_back(i, it.col(), -sumWeights(it.row())*weights(j)*it.value()/delta); }
      }
    }
    b(i) += weights.size()*delta;
  }

}

Eigen::SparseMatrix<double> SparsePenaltyFunction::Hessian(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) {
  assert(weights.size()==outdim);
  std::vector<Eigen::Triplet<double> > entries;
  HessianEntries(beta, weights, entries);

  Eigen::SparseMatrix<double> hess(indim, indim);
  hess.setFromTriplets(entries.begin(), entries.end());
  return hess;
}

Eigen::SparseMatrix<double> SparsePenaltyFunction::HessianFD(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) {
  std::vector<Eigen::Triplet<double> > entries;
  HessianEntriesFD(beta, weights, entries);

  Eigen::SparseMatrix<double> hess(indim, indim);
  hess.setFromTriplets(entries.begin(), entries.end());
  return hess;
}
