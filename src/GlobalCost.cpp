#include "clf/GlobalCost.hpp"

namespace pt = boost::property_tree;
using namespace clf;

GlobalCost::GlobalCost(std::shared_ptr<SupportPointCloud> const& cloud, pt::ptree const& pt) :
SparseCostFunction(cloud->numCoefficients, NumCostFunctions(cloud)),
dofIndices(cloud)
{}

std::size_t GlobalCost::NumCostFunctions(std::shared_ptr<SupportPointCloud> const& cloud) {
  std::size_t num = 0;
  for( auto it=cloud->Begin(); it!=cloud->End(); ++it ) {
    // add the outputs from the uncoupled cost
    num += (*it)->uncoupledCost->valDim;

    // add the outputs from the coupled cost
    for( const auto& coupled : (*it)->coupledCost ) { num += coupled->valDim; }
  }

  return num;
}

std::shared_ptr<UncoupledCost> GlobalCost::GetUncoupledCost(std::size_t const i) const {
  if( i>=dofIndices.cloud->NumSupportPoints() ) { return nullptr; }

  return dofIndices.cloud->GetSupportPoint(i)->uncoupledCost;
}

std::vector<std::shared_ptr<CoupledCost> > GlobalCost::GetCoupledCost(std::size_t const i) const {
  if( i>=dofIndices.cloud->NumSupportPoints() ) { return std::vector<std::shared_ptr<CoupledCost> >(); }

  return dofIndices.cloud->GetSupportPoint(i)->coupledCost;
}

Eigen::VectorXd GlobalCost::CostImpl(Eigen::VectorXd const& beta) const {
  Eigen::VectorXd cost(valDim);
  std::size_t ind = 0;
  for( std::size_t i=0; i<dofIndices.cloud->NumSupportPoints(); ++i ) {
    auto point = dofIndices.cloud->GetSupportPoint(i);

    // compute the uncoupled cost
    cost.segment(ind, point->uncoupledCost->valDim) = point->uncoupledCost->Cost( beta.segment(dofIndices.globalDoFIndices[i], point->NumCoefficients()));
    ind += point->uncoupledCost->valDim;

    // compute the coupled cost
    for( const auto& coupled : point->coupledCost ) {
      auto neigh = coupled->GetNeighbor();
      cost.segment(ind, coupled->valDim) = coupled->ComputeCost(beta.segment(dofIndices.globalDoFIndices[i], point->NumCoefficients()), beta.segment(dofIndices.globalDoFIndices[neigh->GlobalIndex()], neigh->NumCoefficients()));
      ind += coupled->valDim;
    }
  }

  return cost;
}

void GlobalCost::JacobianImpl(Eigen::VectorXd const& beta, Eigen::SparseMatrix<double>& jac) const {
  std::vector<Eigen::Triplet<double> > triplets;
  std::size_t ind = 0;
  for( std::size_t i=0; i<dofIndices.cloud->NumSupportPoints(); ++i ) {
    auto point = dofIndices.cloud->GetSupportPoint(i);

    // compute the uncoupled cost entries
    std::vector<Eigen::Triplet<double> > localTriplets;
    point->uncoupledCost->JacobianTriplets(beta.segment(dofIndices.globalDoFIndices[i], point->NumCoefficients()), localTriplets);
    for( const auto& it : localTriplets ) {
      const std::size_t row = ind+it.row(); assert(row<jac.rows());
      const std::size_t col = dofIndices.globalDoFIndices[i]+it.col(); assert(col<jac.cols());
      triplets.emplace_back(row, col, it.value());
    }
    localTriplets.clear();
    ind += point->uncoupledCost->valDim;

    // compute the coupled cost entries
    for( const auto& coupled : point->coupledCost ) {
      coupled->JacobianTriplets(localTriplets);
      for( const auto& it : localTriplets ) {
        const std::size_t row = ind+it.row(); assert(row<jac.rows());
        const std::size_t col = dofIndices.globalDoFIndices[(it.col()<point->NumCoefficients()? i : coupled->GetNeighbor()->GlobalIndex())] + it.col() - (it.col()<point->NumCoefficients()? 0 : point->NumCoefficients());
        assert(col<jac.cols());
        triplets.emplace_back(row, col, it.value());
      }
      localTriplets.clear();
      ind += coupled->valDim;
    }
  }

  jac.setFromTriplets(triplets.begin(), triplets.end());
}
