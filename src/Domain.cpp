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

Domain::SampleFailure::SampleFailure(std::string const& funcName, std::size_t const nproposals) :
  std::logic_error("CLF Error: " + funcName + " did not propose a valid sample in " + std::to_string(nproposals) + " proposals.")
{}

Domain::SampleFailure::SampleFailure(std::string const& error) :
  std::logic_error("CLF Error: " + error)
{}

Eigen::VectorXd Domain::Sample() {
  Eigen::VectorXd x = ProposeSample();
  if( super ) {
    const std::size_t maxProposed = para->Get<std::size_t>("MaximumProposedSamples", maxProposedSamps_DEFAULT);
    std::size_t nproposed = 0;
    while( !super->Inside(x) && ++nproposed<maxProposed ) { x = ProposeSample(); }
    if( nproposed>=maxProposed ) { throw SampleFailure("Domain::Sample", nproposed); }
  }

  return x;
}

Eigen::VectorXd Domain::SampleBoundary() {
  Eigen::VectorXd x = ProposeBoundarySample();
  if( super ) {
    const std::size_t maxProposed = para->Get<std::size_t>("MaximumProposedSamples", maxProposedSamps_DEFAULT);
    std::size_t nproposed = 0;
    while( !super->Inside(x) && ++nproposed<maxProposed ) { x = ProposeBoundarySample(); }
    if( nproposed>=maxProposed ) { throw SampleFailure("Domain::SampleBoundary", nproposed); }
  }

  return x;
}

Eigen::VectorXd Domain::ProposeSample() {
  throw exceptions::NotImplemented("Domain::ProposeSample");
  return Eigen::VectorXd();
}

Eigen::VectorXd Domain::ProposeBoundarySample() {
  throw exceptions::NotImplemented("Domain::ProposeBoundarySample");
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
