#include "clf/DenseCostFunction.hpp"

using namespace clf;

DenseCostFunction::DenseCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions, std::size_t const outputDimension) : CostFunction(inputDimension, numPenaltyFunctions, outputDimension) {}

DenseCostFunction::DenseCostFunction(std::size_t const inputDimension, std::size_t const numPenaltyFunctions, std::vector<std::pair<std::size_t, std::size_t> > const& outputDimension) : CostFunction(inputDimension, numPenaltyFunctions, outputDimension) {}

void DenseCostFunction::Jacobian(Eigen::VectorXd const& beta, Eigen::MatrixXd& jac) const {
  jac.resize(numPenaltyTerms, inputDimension);
  std::size_t count = 0;
  std::size_t cnt = 0;
  for( std::size_t i=0; i<numPenaltyFunctions; ++i ) {
    if( i>=outputDimension[count].first ) { ++count; }
    jac.block(cnt, 0, outputDimension[count].second, inputDimension) = PenaltyFunctionJacobian(i, beta);
    cnt += outputDimension[count].second;
  }
}

std::vector<Eigen::MatrixXd> DenseCostFunction::PenaltyFunctionHessianByFD(std::size_t const ind, Eigen::VectorXd const& beta, FDOrder const order, double const dbeta) const {
  assert(beta.size()==inputDimension);
  assert(ind<numPenaltyFunctions);
  const std::size_t outputDimension = PenaltyFunctionOutputDimension(ind);
  std::vector<Eigen::MatrixXd> hess(outputDimension, Eigen::MatrixXd(inputDimension, inputDimension));
  
  // precompute if we need this
  Eigen::MatrixXd jac;
  if( order==FDOrder::FIRST_UPWARD | order==FDOrder::FIRST_DOWNWARD ) { jac = PenaltyFunctionJacobian(ind, beta); }
  
  Eigen::VectorXd betaFD = beta;
  for( std::size_t i=0; i<inputDimension; ++i ) {
    switch( order ) {
    case FDOrder::FIRST_UPWARD: {
      betaFD(i) += dbeta;
      const Eigen::MatrixXd jacp = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) -= dbeta;
      for( std::size_t j=0; j<outputDimension; ++j ) { hess[j].row(i) = (jacp.row(j)-jac.row(j))/dbeta; }
      break;
    }
    case FDOrder::FIRST_DOWNWARD: {
      betaFD(i) -= dbeta;
      const Eigen::MatrixXd jacm = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) += dbeta;
      for( std::size_t j=0; j<outputDimension; ++j ) { hess[j].row(i) = (jac.row(j)-jacm.row(j))/dbeta; }
      break;
    }
    case FDOrder::SECOND: {
      betaFD(i) -= dbeta;
      const Eigen::MatrixXd jacm = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) += 2.0*dbeta;
      const Eigen::MatrixXd jacp = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) -= dbeta;
      for( std::size_t j=0; j<outputDimension; ++j ) { hess[j].row(i) = (jacp.row(j)-jacm.row(j))/(2.0*dbeta); }
      break;
    }
    case FDOrder::FOURTH: {
      betaFD(i) -= dbeta;
      const Eigen::MatrixXd jacm1 = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) -= dbeta;
      const Eigen::MatrixXd jacm2 = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) += 3.0*dbeta;
      const Eigen::MatrixXd jacp1 = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) += dbeta;
      const Eigen::MatrixXd jacp2 = PenaltyFunctionJacobian(ind, betaFD);
      betaFD(i) -= 2.0*dbeta;
      for( std::size_t j=0; j<outputDimension; ++j ) { hess[j].row(i) = ((jacm2.row(j)-jacp2.row(j))/12.0+(2.0/3.0)*(jacp1.row(j)-jacm1.row(j)))/dbeta; }
      break;
    }
    case FDOrder::SIXTH: {
      betaFD(i) -= dbeta;
      const Eigen::MatrixXd jacm1 = PenaltyFunctionJacobian(ind, betaFD);
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
      for( std::size_t j=0; j<outputDimension; ++j ) { hess[j].row(i) = ((jacp3.row(j)-jacm3.row(j))/60.0+(3.0/20.0)*(jacm2.row(j)-jacp2.row(j))+(3.0/4.0)*(jacp1.row(j)-jacm1.row(j)))/dbeta; }
      break;
    }
    }
  }
  
  return hess;
}
