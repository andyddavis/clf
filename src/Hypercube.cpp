#include "clf/Hypercube.hpp"

#include <chrono>

using namespace clf;

std::mt19937_64 Hypercube::gen = Hypercube::RandomNumberGenerator();

Hypercube::Hypercube(std::size_t const dim, std::shared_ptr<const Parameters> const& para) :
  Domain(dim, para)
{
  sampler.emplace_back(0.0, 1.0);
  ComputeMapToHypercube();
}

Hypercube::Hypercube(double const left, double const right, std::size_t const dim, std::shared_ptr<const Parameters> const& para) :
  Domain(dim, para)
{
  assert(left<right);
  sampler.emplace_back(left, right);
  ComputeMapToHypercube();
}

Hypercube::Hypercube(std::shared_ptr<const Parameters> const& para) :
  Domain(para->Get<std::size_t>("InputDimension"), para)
{
  std::optional<double> left = para->OptionallyGet<double>("LeftBoundary");
  std::optional<double> right = para->OptionallyGet<double>("RightBoundary");

  if( (bool)left && (bool)right ) {
    assert((*left)<(*right));
    sampler.emplace_back(*left, *right);
  } else {
    sampler.emplace_back(0.0, 1.0);
  }
  
  ComputeMapToHypercube();
}

Hypercube::Hypercube(Eigen::VectorXd const& left, Eigen::VectorXd const& right, std::shared_ptr<const Parameters> const& para) :
  Domain(left.size(), para)
{
  assert(left.size()==dim); assert(right.size()==dim);
  sampler.reserve(dim);
  for( std::size_t i=0; i<dim; ++i ) {
    assert(left(i)<right(i));
    sampler.emplace_back(left(i), right(i));
  }

  ComputeMapToHypercube();
}

std::mt19937_64 Hypercube::RandomNumberGenerator() {
  std::mt19937_64 rng;
  // initialize the random number generator with time-dependent seed
  uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
  rng.seed(ss);

  return rng;
}
  
double Hypercube::LeftBoundary(std::size_t const ind) const { return sampler[std::min(ind, sampler.size()-1)].a(); }

double Hypercube::RightBoundary(std::size_t const ind) const { return sampler[std::min(ind, sampler.size()-1)].b(); }

bool Hypercube::CheckInside(Eigen::VectorXd const& x) const {
  assert(x.size()==dim);
  for( std::size_t i=0; i<dim; ++i ) {
    if( x(i)<LeftBoundary(i) || x(i)>RightBoundary(i) ) { return false; }
  }
  
  return true;
}

void Hypercube::ComputeMapToHypercube() {
  if( sampler.size()==1 ) {
    const double a = LeftBoundary(0), b = RightBoundary(0);
    // we don't need to actually map since this is the domain we want
     if( std::abs(a+1.0)<1.0e-14 && std::abs(b-1.0)<1.0e-14 ) { return; }

     map = std::pair<Eigen::VectorXd, Eigen::VectorXd>(Eigen::VectorXd::Constant(dim, 2.0/(b-a)), -Eigen::VectorXd::Constant(dim, (a+b)/(b-a)));
     return;
  }

  map = std::pair<Eigen::VectorXd, Eigen::VectorXd>(Eigen::VectorXd(dim), Eigen::VectorXd(dim));
  for( std::size_t i=0; i<dim; ++i ) {
    const double a = LeftBoundary(i), b = RightBoundary(i);
    map->first(i) = 2.0/(b-a);
    map->second(i) = -(a+b)/(b-a);
  }
}

Eigen::VectorXd Hypercube::ProposeSample() {
  Eigen::VectorXd samp(dim);
  for( std::size_t i=0; i<dim; ++i ) { samp(i) = sampler[std::min(i, sampler.size()-1)](gen); }
  return samp;
}
