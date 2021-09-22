#include "clf/UncoupledCost.hpp"

#include "clf/SupportPoint.hpp"

namespace pt = boost::property_tree;
using namespace clf;

UncoupledCost::UncoupledCost(std::shared_ptr<SupportPoint> const& point, pt::ptree const& pt) :
SparseCostFunction(point->NumCoefficients(), point->NumNeighbors()*point->model->outputDimension + (pt.get<double>("RegularizationParameter", 0.0)<1.0e-14? 0.0 : point->NumCoefficients())),
point(point),
uncoupledScale(std::sqrt(pt.get<double>("UncoupledScale", 1.0))),
regularizationScale(std::sqrt(pt.get<double>("RegularizationParameter", 0.0)))
{}

double UncoupledCost::UncoupledScale() const { return uncoupledScale*uncoupledScale; }

double UncoupledCost::RegularizationScale() const { return regularizationScale*regularizationScale; }

double UncoupledCost::PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const {
  // get the support point
  auto pnt = point.lock();
  assert(pnt);

  assert(false);
  return 0.0;
}

/*Eigen::VectorXd UncoupledCost::CostImpl(Eigen::VectorXd const& coefficients) const {
  // get the support point
  auto pnt = point.lock();
  assert(pnt);

  // get the kernel evaluation
  const Eigen::VectorXd kernel = uncoupledScale*pnt->NearestNeighborKernel().array().sqrt();
  assert(pnt->NumNeighbors()==kernel.size());

  // loop through the neighbors
  Eigen::VectorXd cost(numPenaltyFunctions);
  for( std::size_t i=0; i<pnt->NumNeighbors(); ++i ) {
    // the location of the neighbor
    std::shared_ptr<SupportPoint> neigh = pnt->NearestNeighbor(i);

    // evaluate the difference between model operator and the right hand side
    cost.segment(i*pnt->model->outputDimension, pnt->model->outputDimension) = kernel(i)*(neigh->model->Operator(neigh->x, coefficients, pnt->GetBasisFunctions()) - neigh->model->RightHandSide(neigh->x));
  }

  // if we are regularizing, these are seperate cost functions
  if( regularizationScale>=1.0e-14 ) { cost.tail(inputDimension) = regularizationScale*coefficients; }

  return cost;
}*/

/*void UncoupledCost::JacobianTriplets(Eigen::VectorXd const& coefficients, std::vector<Eigen::Triplet<double> >& triplets) const {
  // get the support point
  auto pnt = point.lock();
  assert(pnt);

  // get the kernel evaluation
  const Eigen::VectorXd kernel = uncoupledScale*pnt->NearestNeighborKernel().array().sqrt();
  assert(pnt->NumNeighbors()==kernel.size());

  // loop through the neighbors
  for( std::size_t i=0; i<pnt->NumNeighbors(); ++i ) {
    // the location of the neighbor
    std::shared_ptr<SupportPoint> neigh = pnt->NearestNeighbor(i);

    // the Jacobian of the operator
    const Eigen::MatrixXd modelJac = neigh->model->OperatorJacobian(neigh->x, coefficients, pnt->GetBasisFunctions());
    assert(modelJac.rows()==pnt->model->outputDimension);
    assert(modelJac.cols()==inputDimension);

    for( std::size_t d=0; d<pnt->model->outputDimension; ++d ) {
      for( std::size_t j=0; j<pnt->NumCoefficients(); ++j ) {
        if( std::abs(modelJac(d, j))>1.0e-14 ) { triplets.emplace_back(i*pnt->model->outputDimension+d, j, kernel(i)*modelJac(d, j)); }
      }
    }
  }

  if( regularizationScale>=1.0e-14 ) { for( std::size_t i=0; i<pnt->NumCoefficients(); ++i ) { triplets.emplace_back(pnt->NumNeighbors()*pnt->model->outputDimension+i, i, regularizationScale); } }
}*/

/*void UncoupledCost::JacobianImpl(Eigen::VectorXd const& coefficients, Eigen::SparseMatrix<double>& jac) const {
  std::vector<Eigen::Triplet<double> > triplets;
  JacobianTriplets(coefficients, triplets);
  jac.setFromTriplets(triplets.begin(), triplets.end());
}*/
