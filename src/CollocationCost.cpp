#include "clf/CollocationCost.hpp"

using namespace clf;

CollocationCost::CollocationCost(std::shared_ptr<SupportPoint> const& supportPoint, std::vector<std::shared_ptr<CollocationPoint> > const& collocationPoints) :
DenseCostFunction(supportPoint->NumCoefficients(), collocationPoints.size(), (collocationPoints.size()==0? 0 : collocationPoints[0]->model->outputDimension)),
collocationPoints(collocationPoints)
{}

void CollocationCost::ComputeOptimalCoefficients(Eigen::MatrixXd const& data) const {
  /*assert(data.rows()==collocationCloud->OutputDimension());
  assert(data.cols()==collocationCloud->supportCloud->NumPoints());

  for( auto it=collocationCloud->supportCloud->Begin(); it!=collocationCloud->supportCloud->End(); ++it ) {
    auto point = std::dynamic_pointer_cast<SupportPoint>(*it);
    assert(point);

    // get the global indices of this support point's nearest neighbors
    const std::vector<std::size_t>& globalIndices = point->GlobalNeighborIndices();

    // get the local data for the linear system to compute the optimal coefficients
    Eigen::MatrixXd localData(data.rows(), globalIndices.size());
    for( std::size_t i=0; i<globalIndices.size(); ++i ) { localData.col(i) = data.col(globalIndices[i]); }

    point->ComputeOptimalCoefficients(localData);
    }*/
}

Eigen::VectorXd CollocationCost::ComputeCost(Eigen::MatrixXd const& data) const {
  /*// compute the optimal coefficients for each support point
  ComputeOptimalCoefficients(data);

  Eigen::VectorXd cost(numPenaltyFunctions);
  for( std::size_t i=0; i<collocationCloud->numCollocationPoints; ++i ) {
    auto pnt = collocationCloud->GetCollocationPoint(i);
    cost.segment(i*collocationCloud->OutputDimension(), collocationCloud->OutputDimension()) = pnt->Operator() - pnt->RightHandSide();
  }

  return cost;*/
}

Eigen::VectorXd CollocationCost::PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const {
  assert(ind<collocationPoints.size());
  assert(collocationPoints[ind]);

  return std::sqrt(collocationPoints[ind]->weight)*(collocationPoints[ind]->Operator(collocationPoints[ind]->x, beta) - collocationPoints[ind]->model->RightHandSide(collocationPoints[ind]->x));
}

Eigen::MatrixXd CollocationCost::PenaltyFunctionJacobianImpl(std::size_t const ind, Eigen::VectorXd const& beta) const {
  assert(ind<collocationPoints.size());
  assert(collocationPoints[ind]);

  return std::sqrt(collocationPoints[ind]->weight)*collocationPoints[ind]->OperatorJacobian(collocationPoints[ind]->x, beta);
}

bool CollocationCost::IsQuadratic() const {
  // the cost function is quadratic as long as all of the models are linear
  for( const auto& it : collocationPoints ) { if( !it->model->IsLinear() ) { return false ; } }
  return true;
}
