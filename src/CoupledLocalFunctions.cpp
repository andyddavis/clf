#include "clf/CoupledLocalFunctions.hpp"

#include "clf/Hypercube.hpp"

using namespace clf;

CoupledLocalFunctions::CoupledLocalFunctions(std::shared_ptr<MultiIndexSet> const& set, std::shared_ptr<BasisFunctions> const& basis, std::shared_ptr<Domain> const& domain, Eigen::VectorXd const& delta, std::shared_ptr<Parameters> const& para) :
  domain(domain),
  cloud(std::make_shared<PointCloud>(domain))
{
  // sample the support points from the domain
  cloud->AddPoints(para->Get<std::size_t>("NumSupportPoints"));
  for( std::size_t i=0; i<cloud->NumPoints(); ++i ) {
    auto pt = cloud->Get(i);

    // create the local domain
    auto localDomain = std::make_shared<Hypercube>(pt->x-delta, pt->x+delta);
    localDomain->SetSuperset(domain);

    // add the function associated with this support point
    functions[pt->id] = std::make_shared<LocalFunction>(set, basis, localDomain, para);
  }
}

void CoupledLocalFunctions::SetBoundaryCondition(std::shared_ptr<SystemOfEquations> const& system, std::function<bool(std::pair<Eigen::VectorXd, Eigen::VectorXd> const&)> const& func, std::size_t const numPoints) {
  for( std::size_t i=0; i<numPoints; ++i ) {
    // sample a point on the boundary
    const std::pair<Eigen::VectorXd, Eigen::VectorXd> pnt = domain->SampleBoundary(func);

    // find the closest support point
    const std::pair<std::size_t, double> nearest = cloud->ClosestPoint(pnt.first);

    // find the penalty functions
    auto neigh = cloud->Get(nearest.first);
    BoundaryConditions& it = boundaryConditions[neigh->id];

    // see if a boundary condition for this system has already been added
    auto jt = std::upper_bound(it.begin(), it.end(), system, [](std::shared_ptr<SystemOfEquations> const& s, std::shared_ptr<BoundaryCondition> const& b) { return s->id<=b->SystemID(); });
    // if it has not, then create one
    if( jt==it.end() || (*jt)->SystemID()!=system->id ) { jt = it.insert(jt, std::make_shared<BoundaryCondition>(functions[cloud->Get(nearest.first)->id], system)); }
    assert(jt!=it.end());

    // add the boundary point to the point cloud for this boundary cost function
    (*jt)->AddPoint(pnt);
  }
}

void CoupledLocalFunctions::RemoveBoundaryCondition(std::size_t const systemID) {
}

std::size_t CoupledLocalFunctions::NumLocalFunctions() const { return cloud->NumPoints(); }

std::optional<CoupledLocalFunctions::BoundaryConditions> CoupledLocalFunctions::GetBCs(std::size_t const ind) const {
  auto it = boundaryConditions.find(cloud->Get(ind)->id);
  if( it==boundaryConditions.end() ) { return std::nullopt; }
  return it->second;
}
