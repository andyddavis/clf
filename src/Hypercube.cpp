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

Hypercube::Hypercube(std::vector<bool> const& periodic, double const left, double const right, std::shared_ptr<const Parameters> const& para) :
  Domain(periodic.size(), para),
  periodic(periodic),
  hasPeriodicBoundary(std::find(periodic.begin(), periodic.end(), true)!=periodic.end()),
  hasNonPeriodicBoundary(std::find(periodic.begin(), periodic.end(), false)!=periodic.end())
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

Hypercube::Hypercube( std::vector<bool> const& periodic, std::shared_ptr<const Parameters> const& para) :
  Domain(periodic.size(), para),
  periodic(periodic),
  hasPeriodicBoundary(std::find(periodic.begin(), periodic.end(), true)!=periodic.end()),
  hasNonPeriodicBoundary(std::find(periodic.begin(), periodic.end(), false)!=periodic.end())
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

Hypercube::Hypercube(Eigen::VectorXd const& left, Eigen::VectorXd const& right, std::vector<bool> const& periodic, std::shared_ptr<const Parameters> const& para) :
  Domain(left.size(), para),
  periodic(periodic),
  hasPeriodicBoundary(std::find(periodic.begin(), periodic.end(), true)!=periodic.end()),
  hasNonPeriodicBoundary(std::find(periodic.begin(), periodic.end(), false)!=periodic.end())
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

  // check if we have a hypercube as a super set
  auto hypersuper = std::dynamic_pointer_cast<Hypercube>(super);
  
  for( std::size_t i=0; i<dim; ++i ) {
    if( periodic && periodic->at(i) ) { continue; }

    if( hypersuper && hypersuper->Periodic(i) ) {
      const double length = hypersuper->RightBoundary(i) - hypersuper->LeftBoundary(i);
      assert(length>0.0);

      const double boundl = LeftBoundary(i), boundr = RightBoundary(i);

      double check = x(i);
      while( check>boundr ) { check -= length; }
      while( check<boundl ) { check += length; }
      if( check<boundl || check>boundr ) { return false; } else { continue; }
    }

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

  // check if we have a hypercube as a super set
  auto hypersuper = std::dynamic_pointer_cast<Hypercube>(super);
  if( hypersuper ) {
    const std::optional<Eigen::VectorXd> s = hypersuper->MapPeriodic(samp);
    if( s ) { return *s; }
  }

  return samp;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> Hypercube::ProposeBoundarySample() {
  // if ALL of the boundaries are periodic, this is a problem
  if( !hasNonPeriodicBoundary ) { throw SampleFailure("Tried to sample from the boundary of a clf::Hypercube but all coordinate directions are periodic. There is no boundary."); }
  
  // randomly choose and index
  std::size_t ind = rand()%dim; assert(ind<dim);
  while( Periodic(ind) ) { ind = rand()%dim; assert(ind<dim); }
  
  Eigen::VectorXd samp(dim);
  static std::uniform_real_distribution<double> dist(0.0, 1.0);
  const bool left = dist(gen)<0.5;
  samp(ind) = (left? LeftBoundary(ind) : RightBoundary(ind));
  for( std::size_t i=0; i<dim; ++i ) {
    if( i==ind ) { continue; }
    samp(i) = sampler[std::min(i, sampler.size()-1)](gen);
  }

  Eigen::VectorXd normal = Eigen::VectorXd::Zero(dim);
  normal(ind) = (left? -1.0 : 1.0);

  // check if we have a hypercube as a super set
  auto hypersuper = std::dynamic_pointer_cast<Hypercube>(super);
  if( hypersuper ) {
    const std::optional<Eigen::VectorXd> s = hypersuper->MapPeriodic(samp);
    if( s ) { return std::pair<Eigen::VectorXd, Eigen::VectorXd>(*s, normal); }
  }

  return std::pair<Eigen::VectorXd, Eigen::VectorXd>(samp, normal);
}

double Hypercube::Distance(Eigen::VectorXd const& x1, Eigen::VectorXd const& x2) const {
  // we have a super set, use that distance
  if( super ) { return super->Distance(x1, x2); }

  // this is not a periodic domain
  if( !periodic ) { return (x1-x2).norm(); }

  // compute the distance in a periodic domain
  double dist = 0.0;
  for( std::size_t i=0; i<dim; ++i ) {
    double diff;
    if( periodic->at(i) ) {
      const double mn = std::min(x1(i), x2(i)), mx = std::max(x1(i), x2(i));
      diff = std::min(mx-mn, RightBoundary(i)-mx+mn-LeftBoundary(i));
      assert(diff>-1.0e-10);
    } else {
      diff = x1(i)-x2(i);
    }
    dist += diff*diff;
  }
  return std::sqrt(dist);
}

double Hypercube::MapPeriodicCoordinate(std::size_t const ind, double x) const {
  if( !Periodic(ind) ) { return x; }

  const double l = LeftBoundary(ind), r = RightBoundary(ind);
  const double length = r-l; assert(length>0.0);
  while( x<l ) { x += length; }
  while( x>r ) { x -= length; }
  return x;
}

bool Hypercube::Periodic(std::size_t const ind) const {
  if( periodic ) { return periodic->at(ind); }
  return false;
}

bool Hypercube::Periodic() const { return hasPeriodicBoundary; }

std::optional<Eigen::VectorXd> Hypercube::MapPeriodic(Eigen::VectorXd const& x) const {
  // check if we have a hypercube as a super set
  auto hypersuper = std::dynamic_pointer_cast<Hypercube>(super);

  const bool periodic = Periodic();

  // if the domain is not periodic and the super set is not valid or not periodic
  if( !periodic && ( !hypersuper || !hypersuper->Periodic() ) ) { return std::nullopt; }

  Eigen::VectorXd y(dim);
  for( std::size_t i=0; i<dim; ++i ) {
    // if the domain is periodic, then map into the coordinate
    if( periodic ) {
      y(i) = MapPeriodicCoordinate(i, x(i));
      continue;
    }

    y(i) = x(i);

    if( hypersuper->Periodic(i) ) {
      // if it is inside 
      const double boundl = LeftBoundary(i), boundr = RightBoundary(i);
      if( boundl<y(i) && y(i)<boundr ) {  continue; }
      
      // center and length
      const double length = hypersuper->RightBoundary(i)-hypersuper->LeftBoundary(i), center = (boundr+boundl)/2.0;
      
      while( std::abs(y(i)-center)>length/2.0 ) { y(i) += ( y(i)<center? length : -length ); }
    }
  }
  
  return y;
}

Eigen::VectorXd Hypercube::MapToHypercube(Eigen::VectorXd const& x) const {
  const std::optional<Eigen::VectorXd> y = MapPeriodic(x);
  return Domain::MapToHypercube((y? *y : x));
}
