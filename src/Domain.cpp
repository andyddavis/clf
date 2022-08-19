#include "clf/Domain.hpp"

#include "clf/CLFExceptions.hpp"

#include "clf/FiniteDifference.hpp"

using namespace clf;

Domain::Domain(std::size_t const dim, std::shared_ptr<const Parameters> const& para) :
  dim(dim), para(para)
{}

bool Domain::Inside(Eigen::VectorXd const& x) const {
  bool inside = CheckInside(x);
  if( super ) { inside = (inside && super->Inside(x)); }
  return inside;
}

bool Domain::CheckInside(Eigen::VectorXd const& x) const {
   throw exceptions::NotImplemented("Domain::CheckInside");
   return false;
}

Eigen::VectorXd Domain::MapToHypercube(Eigen::VectorXd const& x) const {
  if( map ) { return map->first.asDiagonal()*x + map->second; }
  return x;
}

Eigen::VectorXd Domain::MapToHypercubeJacobian() const {
  if( map ) { return map->first; }
  return Eigen::VectorXd::Ones(dim);
}

Domain::SampleFailure::SampleFailure(std::size_t const nproposals) :
  std::logic_error("CLF Error: Domain::Sample did not propose a valid sample in " + std::to_string(nproposals) + " proposals.")
{}

Eigen::VectorXd Domain::Sample() {
  Eigen::VectorXd x = ProposeSample();
  if( super ) {
    const std::size_t maxProposed = para->Get<std::size_t>("MaximumProposedSamples", maxProposedSamps_DEFAULT);
    std::size_t nproposed = 0;
    while( !super->Inside(x) && ++nproposed<maxProposed ) { x = ProposeSample(); }
    if( nproposed>=maxProposed ) { throw SampleFailure(nproposed); }
  }

  return x;
}

Eigen::VectorXd Domain::ProposeSample() {
  throw exceptions::NotImplemented("Domain::ProposeSample");
  return Eigen::VectorXd();
}

double Domain::Distance(Eigen::VectorXd const& x1, Eigen::VectorXd const& x2) const {
  if( super ) { return super->Distance(x1, x2); }
  return (x1-x2).norm();
}

void Domain::SetSuperset(std::shared_ptr<Domain> const& supset) {
  assert(supset->dim==dim);
  super = supset;
}
