#include "clf/CollocationCost.hpp"

using namespace clf;

CollocationCost::CollocationCost(std::shared_ptr<SupportPoint> const& supportPoint, std::vector<std::shared_ptr<CollocationPoint> > const& collocationPoints) :
SparseCostFunction(supportPoint->NumCoefficients(), collocationPoints.size(), (collocationPoints.size()==0? 0 : collocationPoints[0]->model->outputDimension))
//collocationCloud(collocationCloud)
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
  assert(false);
  return Eigen::VectorXd();
}

Eigen::VectorXd CollocationCost::CostImpl(Eigen::VectorXd const& data) const {
  // compute the optimal coefficients for each support point
  //return ComputeCost(Eigen::Map<const Eigen::MatrixXd>(data.data(), collocationCloud->OutputDimension(), collocationCloud->supportCloud->NumPoints()));
  return Eigen::VectorXd();
}

void CollocationCost::JacobianImpl(Eigen::VectorXd const& data, Eigen::SparseMatrix<double>& jac) const {
  /*Eigen::Map<const Eigen::MatrixXd> dataMat(data.data(), collocationCloud->OutputDimension(), collocationCloud->supportCloud->NumPoints());

  // compute the optimal coefficients for each support point
  ComputeOptimalCoefficients(dataMat);

  const std::size_t outdim = collocationCloud->OutputDimension();

  std::vector<Eigen::Triplet<double> > triplets;
  for( std::size_t i=0; i<collocationCloud->numCollocationPoints; ++i ) {
    // the colocation and the corresponding support point
    auto pnt = collocationCloud->GetCollocationPoint(i);
    assert(pnt);
    auto support = pnt->supportPoint.lock();
    assert(support);

    const Eigen::MatrixXd localJac = pnt->OperatorJacobian();*support->lsJacobian;
    assert(localJac.rows()==outdim);
    assert(localJac.cols()==outdim*support->NumNeighbors());

    const std::vector<std::size_t>& globalIndices = support->GlobalNeighborIndices();
    for( std::size_t locali=0; locali<localJac.rows(); ++locali ) {
      const std::size_t globali = outdim*i + locali;
      assert(globali<jac.rows());
      for( std::size_t j=0; j<support->NumNeighbors(); ++j ) {
        for( std::size_t d=0; d<outdim; ++d ) {
          const std::size_t localj = j*outdim+d;
          assert(localj<localJac.cols());
          const std::size_t globalj = outdim*globalIndices[j]+d;
          assert(globalj<jac.cols());
          if( std::abs(localJac(locali, localj))>1.0e-14 ) {
            triplets.emplace_back(globali, globalj, localJac(locali, localj));
          }
        }
      }
    }
  }

  jac.setFromTriplets(triplets.begin(), triplets.end());*/
}

bool CollocationCost::IsQuadratic() const {
  assert(false);
  return false;
}
