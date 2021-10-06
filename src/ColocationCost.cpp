#include "clf/ColocationCost.hpp"

using namespace clf;

ColocationCost::ColocationCost(std::shared_ptr<ColocationPointCloud> const& colocationCloud) :
SparseCostFunction(colocationCloud->OutputDimension()*colocationCloud->supportCloud->NumPoints(), colocationCloud->OutputDimension()*colocationCloud->numColocationPoints, 1),
colocationCloud(colocationCloud)
{}

void ColocationCost::ComputeOptimalCoefficients(Eigen::MatrixXd const& data) const {
  assert(data.rows()==colocationCloud->OutputDimension());
  assert(data.cols()==colocationCloud->supportCloud->NumPoints());

  for( auto it=colocationCloud->supportCloud->Begin(); it!=colocationCloud->supportCloud->End(); ++it ) {
    auto point = std::dynamic_pointer_cast<SupportPoint>(*it);
    assert(point);

    // get the global indices of this support point's nearest neighbors
    const std::vector<std::size_t>& globalIndices = point->GlobalNeighborIndices();

    // get the local data for the linear system to compute the optimal coefficients
    Eigen::MatrixXd localData(data.rows(), globalIndices.size());
    for( std::size_t i=0; i<globalIndices.size(); ++i ) { localData.col(i) = data.col(globalIndices[i]); }

    point->ComputeOptimalCoefficients(localData);
  }
}

Eigen::VectorXd ColocationCost::ComputeCost(Eigen::MatrixXd const& data) const {
  // compute the optimal coefficients for each support point
  ComputeOptimalCoefficients(data);

  Eigen::VectorXd cost(numPenaltyFunctions);
  for( std::size_t i=0; i<colocationCloud->numColocationPoints; ++i ) {
    auto pnt = colocationCloud->GetColocationPoint(i);
    cost.segment(i*colocationCloud->OutputDimension(), colocationCloud->OutputDimension()) = pnt->Operator() - pnt->RightHandSide();
  }

  return cost;
}

Eigen::VectorXd ColocationCost::PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const {
  assert(false);
  return Eigen::VectorXd();
}

Eigen::VectorXd ColocationCost::CostImpl(Eigen::VectorXd const& data) const {
  // compute the optimal coefficients for each support point
  return ComputeCost(Eigen::Map<const Eigen::MatrixXd>(data.data(), colocationCloud->OutputDimension(), colocationCloud->supportCloud->NumPoints()));
}

void ColocationCost::JacobianImpl(Eigen::VectorXd const& data, Eigen::SparseMatrix<double>& jac) const {
  Eigen::Map<const Eigen::MatrixXd> dataMat(data.data(), colocationCloud->OutputDimension(), colocationCloud->supportCloud->NumPoints());

  // compute the optimal coefficients for each support point
  ComputeOptimalCoefficients(dataMat);

  const std::size_t outdim = colocationCloud->OutputDimension();

  std::vector<Eigen::Triplet<double> > triplets;
  for( std::size_t i=0; i<colocationCloud->numColocationPoints; ++i ) {
    // the colocation and the corresponding support point
    auto pnt = colocationCloud->GetColocationPoint(i);
    assert(pnt);
    auto support = pnt->supportPoint.lock();
    assert(support);

    const Eigen::MatrixXd localJac = pnt->OperatorJacobian()*support->lsJacobian;
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

  jac.setFromTriplets(triplets.begin(), triplets.end());
}

bool ColocationCost::IsQuadratic() const {
  assert(false);
  return false;
}
