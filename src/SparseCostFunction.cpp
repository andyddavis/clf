#include "clf/SparseCostFunction.hpp"

using namespace clf;

SparseCostFunction::SparseCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions, std::size_t const outputDimension) : CostFunction(inputDimension, numPenaltyFunctions, outputDimension) {}

std::vector<Eigen::Triplet<double> > SparseCostFunction::PenaltyFunctionJacobianSparse(std::size_t const ind, Eigen::VectorXd const& beta) const {
  assert(beta.size()==inputDimension);
  assert(ind<numPenaltyFunctions);
  const std::vector<Eigen::Triplet<double> > jac = PenaltyFunctionJacobianSparseImpl(ind, beta);
  return jac;
}

void SparseCostFunction::Jacobian(Eigen::VectorXd const& beta, Eigen::SparseMatrix<double>& jac) const {
  // resize the jacobian---this sets every entry to zero, but does not free the memory
  jac.resize(numPenaltyTerms, inputDimension);
  std::vector<Eigen::Triplet<double> > triplets;
  std::size_t count = 0;
  std::size_t cnt = 0;
  for( std::size_t i=0; i<numPenaltyFunctions; ++i ) {
    if( i>=outputDimension[count].first ) { ++count; }
    const std::vector<Eigen::Triplet<double> > jaci = PenaltyFunctionJacobianSparse(i, beta);
    for( const auto& it : jaci ) { triplets.emplace_back(cnt+it.row(), it.col(), it.value()); }
    cnt += outputDimension[count].second;
  }
  jac.setFromTriplets(triplets.begin(), triplets.end());
  jac.makeCompressed();
}

Eigen::MatrixXd SparseCostFunction::PenaltyFunctionJacobianImpl(std::size_t const ind, Eigen::VectorXd const& beta) const {
  const std::vector<Eigen::Triplet<double> > sparseJac = PenaltyFunctionJacobianSparseImpl(ind, beta);

  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(PenaltyFunctionOutputDimension(ind), inputDimension);
  for( const auto& it : sparseJac ) { jac(it.row(), it.col()) = it.value(); }
  return jac;
}

std::vector<Eigen::Triplet<double> > SparseCostFunction::PenaltyFunctionJacobianSparseImpl(std::size_t const ind, Eigen::VectorXd const& beta) const {
  const std::size_t outputDimension = PenaltyFunctionOutputDimension(ind);
  const Eigen::MatrixXd jac = PenaltyFunctionJacobianByFD(ind, beta);
  std::vector<Eigen::Triplet<double> > sparseJac;
  for( std::size_t i=0; i<outputDimension; ++i ) {
    for( std::size_t j=0; j<inputDimension; ++j ) {
      if( std::abs(jac(i, j))>sparsityTol ) { sparseJac.emplace_back(i, j, jac(i, j)); }
    }
  }
  return sparseJac;
}

std::vector<Eigen::SparseMatrix<double> > SparseCostFunction::PenaltyFunctionHessianByFD(std::size_t const ind, Eigen::VectorXd const& beta, FDOrder const order, double const dbeta) const {
  assert(beta.size()==inputDimension);
  assert(ind<numPenaltyFunctions);
  const std::size_t outputDimension = PenaltyFunctionOutputDimension(ind);
  std::vector<Eigen::SparseMatrix<double> > hess(outputDimension, Eigen::SparseMatrix<double>(inputDimension, inputDimension));

  // precompute if we need this
  Eigen::MatrixXd jac;
  if( order==FDOrder::FIRST_UPWARD | order==FDOrder::FIRST_DOWNWARD ) { jac = PenaltyFunctionJacobian(ind, beta); }

  std::vector<std::vector<Eigen::Triplet<double> > > entries(outputDimension);
  Eigen::VectorXd betaFD = beta;
  for( std::size_t i=0; i<inputDimension; ++i ) {
    switch( order ) {
    case FDOrder::FIRST_UPWARD: {
      betaFD(i) += dbeta;
      const Eigen::MatrixXd deriv = (PenaltyFunctionJacobian(ind, betaFD)-jac)/dbeta;
      betaFD(i) -= dbeta;
      for( std::size_t j1=0; j1<outputDimension; ++j1 ) { 
	for( std::size_t j2=0; j2<inputDimension; ++j2 ) { 
	  if( std::abs(deriv(j1, j2))>sparsityTol ) { entries[j1].emplace_back(i, j2, deriv(j1, j2)); }
	}
      }
      break;
    }
    case FDOrder::FIRST_DOWNWARD: {
      betaFD(i) -= dbeta;
      const Eigen::MatrixXd deriv = (jac-PenaltyFunctionJacobian(ind, betaFD))/dbeta;
      betaFD(i) += dbeta;
      for( std::size_t j1=0; j1<outputDimension; ++j1 ) { 
	for( std::size_t j2=0; j2<inputDimension; ++j2 ) { 
	  if( std::abs(deriv(j1, j2))>sparsityTol ) { entries[j1].emplace_back(i, j2, deriv(j1, j2)); }
	}
      }
      break;
    }
    case FDOrder::SECOND: {
      betaFD(i) -= dbeta;
      Eigen::MatrixXd jacm = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) += 2.0*dbeta;
      const Eigen::MatrixXd jacp = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) -= dbeta;

      jacm = (jacp-jacm)/(2.0*dbeta);
      for( std::size_t j1=0; j1<outputDimension; ++j1 ) { 
	for( std::size_t j2=0; j2<inputDimension; ++j2 ) { 
	  if( std::abs(jacm(j1, j2))>sparsityTol ) { entries[j1].emplace_back(i, j2, jacm(j1, j2)); }
	}
      }
      break;
    }
    case FDOrder::FOURTH: {
      betaFD(i) -= dbeta;
      Eigen::MatrixXd jacm1 = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) -= dbeta;
      const Eigen::MatrixXd jacm2 = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) += 3.0*dbeta;
      const Eigen::MatrixXd jacp1 = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) += dbeta;
      const Eigen::MatrixXd jacp2 = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) -= 2.0*dbeta;

      jacm1 = ((jacm2-jacp2)/12.0+(2.0/3.0)*(jacp1-jacm1))/dbeta;
      for( std::size_t j1=0; j1<outputDimension; ++j1 ) { 
	for( std::size_t j2=0; j2<inputDimension; ++j2 ) { 
	  if( std::abs(jacm1(j1, j2))>sparsityTol ) { entries[j1].emplace_back(i, j2, jacm1(j1, j2)); }
	}
      }
      break;
    }
    case FDOrder::SIXTH: {
      betaFD(i) -= dbeta;
      Eigen::MatrixXd jacm1 = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) -= dbeta;
      const Eigen::MatrixXd jacm2 = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) -= dbeta;
      const Eigen::MatrixXd jacm3 = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) += 4.0*dbeta;
      const Eigen::MatrixXd jacp1 = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) += dbeta;
      const Eigen::MatrixXd jacp2 = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) += dbeta;
      const Eigen::MatrixXd jacp3 = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) -= 3.0*dbeta;

      jacm1 = ((jacp3-jacm3)/60.0+(3.0/20.0)*(jacm2-jacp2)+(3.0/4.0)*(jacp1-jacm1))/dbeta;
      for( std::size_t j1=0; j1<outputDimension; ++j1 ) { 
	for( std::size_t j2=0; j2<inputDimension; ++j2 ) { 
	  if( std::abs(jacm1(j1, j2))>sparsityTol ) { entries[j1].emplace_back(i, j2, jacm1(j1, j2)); }
	}
      }
      break;
    }
    }
  }

  for( std::size_t i=0; i<outputDimension; ++i ) { hess[i].setFromTriplets(entries[i].begin(), entries[i].end()); }
  return hess;
}

std::vector<Eigen::SparseMatrix<double> > SparseCostFunction::PenaltyFunctionHessianImpl(std::size_t const ind, Eigen::VectorXd const& beta) const {
  const std::vector<std::vector<Eigen::Triplet<double> > > hess = PenaltyFunctionHessianSparseImpl(ind, beta);
  if( hess.empty() ) { return PenaltyFunctionHessianByFD(ind, beta); }
  const size_t outputDimension = PenaltyFunctionOutputDimension(ind);
  assert(hess.size()==outputDimension);

  std::vector<Eigen::SparseMatrix<double> > hessMat(outputDimension, Eigen::SparseMatrix<double>(inputDimension, inputDimension));
  for( std::size_t i=0; i<outputDimension; ++i ) { hessMat[i].setFromTriplets(hess[i].begin(), hess[i].end()); }
  
  return hessMat;
}

std::vector<std::vector<Eigen::Triplet<double> > > SparseCostFunction::PenaltyFunctionHessianSparseImpl(std::size_t const ind, Eigen::VectorXd const& beta) const { return std::vector<std::vector<Eigen::Triplet<double> > >(); }
