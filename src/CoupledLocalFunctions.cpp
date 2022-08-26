#include "clf/CoupledLocalFunctions.hpp"

#include "clf/Hypercube.hpp"
#include "clf/BoundaryCondition.hpp"
#include "clf/LocalResidual.hpp"
#include "clf/DenseLevenbergMarquardt.hpp"

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

  coefficients = Eigen::VectorXd::Zero(NumCoefficients()); 
}

void CoupledLocalFunctions::AddBoundaryCondition(std::shared_ptr<SystemOfEquations> const& system, std::function<bool(std::pair<Eigen::VectorXd, Eigen::VectorXd> const&)> const& func, std::size_t const numPoints) {
  for( std::size_t i=0; i<numPoints; ++i ) {
    // sample a point on the boundary
    const std::pair<Eigen::VectorXd, Eigen::VectorXd> pnt = domain->SampleBoundary(func);

    // find the closest support point
    const std::pair<std::size_t, double> nearest = cloud->ClosestPoint(pnt.first);

    // find the penalty functions
    auto neigh = cloud->Get(nearest.first);
    Residuals& it = residuals[neigh->id];

    // see if a boundary condition for this system has already been added
    auto jt = std::upper_bound(it.begin(), it.end(), system, [](std::shared_ptr<SystemOfEquations> const& s, std::shared_ptr<Residual> const& b) { return s->id<=b->SystemID(); });
    // if it has not, then create one
    if( jt==it.end() || (*jt)->SystemID()!=system->id ) { jt = it.insert(jt, std::make_shared<BoundaryCondition>(functions[cloud->Get(nearest.first)->id], system)); }
    assert(jt!=it.end());

    // add the boundary point to the point cloud for this boundary cost function
    auto bc = std::dynamic_pointer_cast<BoundaryCondition>(*jt);
    assert(bc);
    bc->AddPoint(pnt);
  }
}

void CoupledLocalFunctions::RemoveResidual(std::size_t const systemID) {
  // loop through all of the residuals
  for( auto it=residuals.begin(); it!=residuals.end(); ) {
    // see this system has been added to this function
    auto jt = std::upper_bound(it->second.begin(), it->second.end(), systemID, [](std::size_t const s, std::shared_ptr<Residual> const& b) { return s<=b->SystemID(); });

    // remove it
    if( jt!=it->second.end() ) { it->second.erase(jt); }

    // if the vector of boundary conditions is size 0, remove it too
    it = ( it->second.size()==0? residuals.erase(it) : ++it );
  }
}

std::size_t CoupledLocalFunctions::NumLocalFunctions() const { return cloud->NumPoints(); }

std::optional<CoupledLocalFunctions::Residuals> CoupledLocalFunctions::GetResiduals(std::size_t const ind) const {
  auto it = residuals.find(cloud->Get(ind)->id);
  if( it==residuals.end() ) { return std::nullopt; }
  return it->second;
}

void CoupledLocalFunctions::AddResidual(std::shared_ptr<SystemOfEquations> const& system, std::shared_ptr<const Parameters> const& para) {
  for( auto it=cloud->Begin(); it!=cloud->End(); ++it ) {
    Residuals& resids = residuals[(*it)->id];
    
    auto jt = std::upper_bound(resids.begin(), resids.end(), system, [](std::shared_ptr<SystemOfEquations> const s, std::shared_ptr<Residual> const& resid) { return s->id<=resid->SystemID(); });
    // it has aready been added
    if( jt!=resids.end() && (*jt)->SystemID()==system->id) { continue; }

    resids.insert(jt, std::make_shared<LocalResidual>(functions[(*it)->id], system, para));
  }
}

void CoupledLocalFunctions::MinimizeResiduals() {
  std::size_t start = 0;
  for( auto it=cloud->Begin(); it!=cloud->End(); ++it ) {
    DensePenaltyFunctions resids(residuals[(*it)->id].begin(), residuals[(*it)->id].end());

    auto lm = std::make_shared<DenseLevenbergMarquardt>(std::make_shared<DenseCostFunction>(resids));

    auto func = functions[(*it)->id];
    Eigen::VectorXd cost;
    auto check = lm->Minimize(coefficients.segment(start, func->NumCoefficients()), cost);
    assert(check.first>0);
    start += func->NumCoefficients();
  }

  std::cout << coefficients.transpose() << std::endl << std::endl;
}

std::size_t CoupledLocalFunctions::NumCoefficients() const {
  std::size_t num = 0;
  for( auto it=functions.begin(); it!=functions.end(); ++it ) { num += it->second->NumCoefficients(); }
  return num;
}

Eigen::VectorXd CoupledLocalFunctions::Evaluate(Eigen::VectorXd const& x) const {
  std::pair<std::size_t, double> closest = cloud->ClosestPoint(x);
  auto pnt = cloud->Get(closest.first);

  std::size_t start = 0;
  for( auto it=cloud->Begin(); it!=cloud->End(); ++it ) {
    if( pnt->id==(*it)->id ) { break; }
    
    auto func = functions.at((*it)->id);
    start += func->NumCoefficients();
  }

  auto func = functions.at(pnt->id);
  return func->Evaluate(x, coefficients.segment(start, func->NumCoefficients()));
}
