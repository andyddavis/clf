#include "clf/GlobalCost.hpp"

namespace pt = boost::property_tree;
using namespace clf;

GlobalCost::GlobalCost(std::shared_ptr<SupportPointCloud> const& cloud, pt::ptree const& pt) :
SparseCostFunction(cloud->numCoefficients, NumCostFunctions(cloud), 1),
dofIndices(cloud)
{}

std::size_t GlobalCost::NumCostFunctions(std::shared_ptr<SupportPointCloud> const& cloud) {
  std::size_t num = 0;
  for( auto it=cloud->Begin(); it!=cloud->End(); ++it ) {
    auto point = std::dynamic_pointer_cast<SupportPoint>(*it);
    assert(point);

    // add the outputs from the uncoupled cost
    num += point->uncoupledCost->numPenaltyFunctions;

    // add the outputs from the coupled cost
    for( const auto& coupled : point->coupledCost ) { num += coupled->numPenaltyFunctions; }
  }

  return num;
}

std::shared_ptr<UncoupledCost> GlobalCost::GetUncoupledCost(std::size_t const i) const {
  if( i>=dofIndices.cloud->NumPoints() ) { return nullptr; }

  return dofIndices.cloud->GetSupportPoint(i)->uncoupledCost;
}

std::vector<std::shared_ptr<CoupledCost> > GlobalCost::GetCoupledCost(std::size_t const i) const {
  if( i>=dofIndices.cloud->NumPoints() ) { return std::vector<std::shared_ptr<CoupledCost> >(); }

  return dofIndices.cloud->GetSupportPoint(i)->coupledCost;
}

Eigen::VectorXd GlobalCost::PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const {
  assert(false);
  return Eigen::VectorXd();
}

Eigen::VectorXd GlobalCost::CostImpl(Eigen::VectorXd const& beta) const {
  Eigen::VectorXd cost(numPenaltyFunctions);
  std::size_t ind = 0;
  for( std::size_t i=0; i<dofIndices.cloud->NumPoints(); ++i ) {
    auto point = dofIndices.cloud->GetSupportPoint(i);

    // compute the uncoupled cost
    cost.segment(ind, point->uncoupledCost->numPenaltyFunctions) = point->uncoupledCost->CostVector(beta.segment(dofIndices.globalDoFIndices[i], point->NumCoefficients()));
    ind += point->uncoupledCost->numPenaltyFunctions;

    // compute the coupled cost
    for( const auto& coupled : point->coupledCost ) {
      auto neigh = coupled->GetNeighbor();
      //cost.segment(ind, coupled->numPenaltyFunctions) = coupled->ComputeCost(beta.segment(dofIndices.globalDoFIndices[i], point->NumCoefficients()), beta.segment(dofIndices.globalDoFIndices[neigh->GlobalIndex()], neigh->NumCoefficients()));
      ind += coupled->numPenaltyFunctions;
    }
  }

  return cost;
}

void GlobalCost::JacobianImpl(Eigen::VectorXd const& beta, Eigen::SparseMatrix<double>& jac) const {
  std::vector<Eigen::Triplet<double> > triplets;
  std::size_t ind = 0;
  for( std::size_t i=0; i<dofIndices.cloud->NumPoints(); ++i ) {
    auto point = dofIndices.cloud->GetSupportPoint(i);

    // compute the uncoupled cost entries
    std::vector<Eigen::Triplet<double> > localTriplets;
    //point->uncoupledCost->JacobianTriplets(beta.segment(dofIndices.globalDoFIndices[i], point->NumCoefficients()), localTriplets);
    for( const auto& it : localTriplets ) {
      const std::size_t row = ind+it.row(); assert(row<jac.rows());
      const std::size_t col = dofIndices.globalDoFIndices[i]+it.col(); assert(col<jac.cols());
      triplets.emplace_back(row, col, it.value());
    }
    localTriplets.clear();
    ind += point->uncoupledCost->numPenaltyFunctions;

    // compute the coupled cost entries
    for( const auto& coupled : point->coupledCost ) {
      //coupled->JacobianTriplets(localTriplets);
      for( const auto& it : localTriplets ) {
        const std::size_t row = ind+it.row(); assert(row<jac.rows());
        const std::size_t col = dofIndices.globalDoFIndices[(it.col()<point->NumCoefficients()? i : coupled->GetNeighbor()->GlobalIndex())] + it.col() - (it.col()<point->NumCoefficients()? 0 : point->NumCoefficients());
        assert(col<jac.cols());
        triplets.emplace_back(row, col, it.value());
      }
      localTriplets.clear();
      ind += coupled->numPenaltyFunctions;
    }
  }

  jac.setFromTriplets(triplets.begin(), triplets.end());
}

bool GlobalCost::IsQuadratic() const {
  assert(false);
  return false;
}
