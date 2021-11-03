#include "clf/SupportPoint.hpp"

#include <MUQ/Optimization/NLoptOptimizer.h>

#include "clf/UtilityFunctions.hpp"
#include "clf/PolynomialBasis.hpp"
#include "clf/SinCosBasis.hpp"
#include "clf/SupportPointCloud.hpp"

#include "clf/LevenbergMarquardt.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::Optimization;
using namespace clf;

SupportPoint::SupportPoint(Eigen::VectorXd const& x, pt::ptree const& pt) :
Point(x)
{}

SupportPoint::SupportPoint(Eigen::VectorXd const& x, std::shared_ptr<const Model> const& model, pt::ptree const& pt) :
Point(x, model)
{}

std::size_t SupportPoint::DetermineNumNeighbors(std::vector<std::shared_ptr<const BasisFunctions> > const& bases, pt::ptree const& pt) {
  std::size_t numNeighs = pt.get<std::size_t>("NumNeighbors", std::numeric_limits<std::size_t>::max());
  if( numNeighs<std::numeric_limits<std::size_t>::max() ) {
    for( std::size_t d=0; d<bases.size(); ++d ) {
      if( numNeighs<bases[d]->NumBasisFunctions() ) { throw exceptions::SupportPointWrongNumberOfNearestNeighbors(d, bases[d]->NumBasisFunctions(), numNeighs); }
    }
    return numNeighs;
  }

  numNeighs = 0;
  for( std::size_t d=0; d<bases.size(); ++d ) { numNeighs = std::max(numNeighs, bases[d]->NumBasisFunctions()+1); }

  return numNeighs;
}

std::vector<std::shared_ptr<const BasisFunctions> > SupportPoint::CreateBasisFunctions(std::shared_ptr<SupportPoint> const& point, std::size_t const indim, std::size_t const outdim, pt::ptree pt) {
  // get the names of each child tree that contains the options for each basis
  std::string basisOptionNames = pt.get<std::string>("BasisFunctions");
  // remove spaces
  basisOptionNames.erase(std::remove(basisOptionNames.begin(), basisOptionNames.end(), ' '), basisOptionNames.end());
  // check for leading or trailing comma
  if( *(basisOptionNames.begin())==',' ) { basisOptionNames.erase(0, 1); }
  if( *(basisOptionNames.end()-1)==',' ) { basisOptionNames.erase(basisOptionNames.size()-1, 1); }

  const std::size_t givendim = std::count(basisOptionNames.begin(), basisOptionNames.end(), ',')+1;
  if( givendim!=outdim ) { throw exceptions::SupportPointWrongNumberOfBasesConstructed(outdim, givendim, basisOptionNames); }

  // create each basis
  std::vector<std::shared_ptr<const BasisFunctions> > bases;
  bases.reserve(outdim);
  while( !basisOptionNames.empty() ) {
    // get the option name for this basis
    const std::size_t pos = basisOptionNames.find(',');
    const std::string basisOptionName = basisOptionNames.substr(0, pos);

    // create the basis
    bases.push_back(CreateBasisFunctions(point, indim, pt.get_child(basisOptionName)));

    // remove this basis from the list of options
    basisOptionNames.erase(0, pos==std::string::npos? pos : pos+1);
  }

  return bases;
}

std::shared_ptr<const BasisFunctions> SupportPoint::CreateBasisFunctions(std::shared_ptr<SupportPoint> const& point, std::size_t const indim, pt::ptree pt) {
  // find the time we are trying to create and make sure it is a valid option
  const std::string type = UtilityFunctions::ToUpper(pt.get<std::string>("Type"));
  if( std::find(exceptions::SupportPointInvalidBasisException::options.begin(), exceptions::SupportPointInvalidBasisException::options.end(), type)==exceptions::SupportPointInvalidBasisException::options.end() ) { throw exceptions::SupportPointInvalidBasisException(type); }

  // create the basis and return it
  std::shared_ptr<const BasisFunctions> basis;
  if( type=="TOTALORDERPOLYNOMIALS" ) {
    pt.put("InputDimension", indim);
    basis = PolynomialBasis::TotalOrderBasis(pt);
  } else if( type=="TOTALORDERSINCOS" ) {
    pt.put("InputDimension", indim);
    basis = SinCosBasis::TotalOrderBasis(pt);
  }

  // check for invalid basis type, throw an exception
  if( !basis ) { throw exceptions::SupportPointInvalidBasisException(type); }

  if( pt.get<bool>("LocalBasis", true) ) { basis = std::make_shared<SupportPointBasis>(point, basis, pt); }

  return basis;
}

std::size_t SupportPoint::ComputeNumCoefficients(std::vector<std::shared_ptr<const BasisFunctions> > const& bases) {
  std::size_t num = 0;
  for( const auto& it : bases ) { num += it->NumBasisFunctions(); }
  return num;
}

std::size_t SupportPoint::NumCoefficients() const { return numCoefficients; }

double SupportPoint::SquaredDistanceToNeighbor(std::size_t const i) const {
  if( i>=squaredNeighborDistances.size() ) { return std::numeric_limits<double>::quiet_NaN(); }

  return squaredNeighborDistances[i];
}

void SupportPoint::SetNearestNeighbors(std::shared_ptr<const SupportPointCloud> const& newcloud, std::vector<unsigned int> const& neighInd, std::vector<double> const& neighDist) {
  assert(neighInd.size()==neighDist.size());
  assert(neighInd.size()==numNeighbors);

  // set the cloud
  cloud = newcloud;

  // sort the nearest neighbors (so j=0 is always this support point)
  std::vector<std::size_t> indices(neighInd.size());
  for( std::size_t i=0; i<indices.size(); ++i ) { indices[i] = i; }
  std::sort(indices.begin(), indices.end(), [neighDist](std::size_t const i, std::size_t const j) { return neighDist[i]<neighDist[j]; } );
  assert(neighDist[indices[0]]<1.0e-12); // the first neighbor should itself
  if( neighDist.size()>1 ) { assert(neighDist[indices[1]]>0.0); } // the other neighbors should be farther away

  squaredNeighborDistances.resize(indices.size());
  globalNeighorIndices.resize(indices.size());
  for( std::size_t i=0; i<indices.size(); ++i ) {
    squaredNeighborDistances[i] = neighDist[indices[i]];
    globalNeighorIndices[i] = neighInd[indices[i]];
  }

  // compute and store the kernel evaluations
  ComputeNearestNeighborKernel();

  // reset the scaling parameter for the squared neighbor distance
  for( const auto& basis : bases ) {
    auto suppBasis = std::dynamic_pointer_cast<const SupportPointBasis>(basis);
    if( suppBasis ) { suppBasis->SetRadius(std::sqrt(*(squaredNeighborDistances.end()-1))); }
  }

  // compute and store information required to solve the least squares problem (with an identity model)
  ComputeLeastSquaresInformation();
}

double SupportPoint::CouplingFunction(std::size_t const neighInd) const { return 0.0; }

void SupportPoint::CreateCoupledCosts() {
  auto cld = cloud.lock();
  assert(cld);

  coupledCost.reserve(globalNeighorIndices.size()-1);
  // create the coupling cost for each neighbor, don't couple with the first neighbor since it is this
  for( std::size_t i=1; i<globalNeighorIndices.size(); ++i ) {
    // we might not need to create coupling costs
    const double couplingScale = CouplingFunction(i);
    if( CouplingFunction(i)<1.0e-12 ) { continue; }

    pt::ptree couplingOptions;
    couplingOptions.put("CoupledScale", couplingScale);

    coupledCost.push_back(std::make_shared<CoupledCost>(shared_from_this(), cld->GetSupportPoint(globalNeighorIndices[i]), couplingOptions));
  }
}

std::size_t SupportPoint::GlobalIndex() const { return (globalNeighorIndices.size()==0? std::numeric_limits<std::size_t>::max() : globalNeighorIndices[0]); }

std::vector<std::size_t> const& SupportPoint::GlobalNeighborIndices() const { return globalNeighorIndices; }

bool SupportPoint::IsNeighbor(std::size_t const& globalInd) const { return std::find(globalNeighorIndices.begin(), globalNeighorIndices.end(), globalInd)!=globalNeighorIndices.end(); }

std::size_t SupportPoint::LocalIndex(std::size_t const globalInd) const {
  const auto localInd = std::find(globalNeighorIndices.begin(), globalNeighorIndices.end(), globalInd);
  if( localInd==globalNeighorIndices.end() ) { return std::numeric_limits<std::size_t>::max(); }
  return std::distance(globalNeighorIndices.begin(), localInd);
}

std::shared_ptr<SupportPoint> SupportPoint::NearestNeighbor(std::size_t const jnd) const {
  // get the cloud
  auto cld = cloud.lock();
  assert(cld);

  assert(jnd<globalNeighorIndices.size());
  return cld->GetSupportPoint(GlobalNeighborIndex(jnd));
}

std::size_t SupportPoint::GlobalNeighborIndex(std::size_t const localInd) const {
  assert(localInd<globalNeighorIndices.size());
  return globalNeighorIndices[localInd];
}

void SupportPoint::ComputeLeastSquaresInformation() {
  Eigen::MatrixXd vand = Eigen::MatrixXd::Zero(NumNeighbors()*model->outputDimension, numCoefficients);
  Eigen::VectorXd kernel(NumNeighbors()*model->outputDimension);

  for( std::size_t i=0; i<NumNeighbors(); ++i ) {
    auto neigh = NearestNeighbor(i);

    std::size_t ind = 0;
    for( std::size_t d=0; d<model->outputDimension; ++d ) {
      vand.block(i*model->outputDimension+d, ind, 1, bases[d]->NumBasisFunctions()) = bases[d]->EvaluateBasisFunctions(neigh->x).transpose();
      ind += bases[d]->NumBasisFunctions();
    }
    kernel.segment(i*model->outputDimension, model->outputDimension) = Eigen::VectorXd::Constant(model->outputDimension, NearestNeighborKernel(i));
  }

  lsJacobian = (vand.transpose()*kernel.asDiagonal()*vand).ldlt().solve(vand.transpose()*kernel.asDiagonal());
}

void SupportPoint::ComputeNearestNeighborKernel() {
  nearestNeighborKernel.resize(squaredNeighborDistances.size());

  for( std::size_t i=0; i<nearestNeighborKernel.size(); ++i ) { nearestNeighborKernel(i) = model->NearestNeighborKernel(squaredNeighborDistances[i]/(*(squaredNeighborDistances.end()-1))); }
}

Eigen::VectorXd SupportPoint::NearestNeighborKernel() const { return nearestNeighborKernel; }

double SupportPoint::NearestNeighborKernel(std::size_t const ind) const {
  assert(ind<nearestNeighborKernel.size());
  return nearestNeighborKernel(ind);
}

Eigen::VectorXd SupportPoint::Operator() const {
  assert(model);
  assert(coefficients.size()==numCoefficients);
  return model->Operator(x, coefficients, bases);
}

Eigen::VectorXd SupportPoint::Operator(Eigen::VectorXd const& loc) const {
  assert(model);
  assert(loc.size()==model->inputDimension);
  assert(coefficients.size()==numCoefficients);
  return model->Operator(loc, coefficients, bases);
}

Eigen::VectorXd SupportPoint::Operator(Eigen::VectorXd const& loc, Eigen::VectorXd const& coeffs) const {
  assert(model);
  assert(loc.size()==model->inputDimension);
  assert(coeffs.size()==numCoefficients);
  return model->Operator(loc, coeffs, bases);
}

Eigen::MatrixXd SupportPoint::OperatorJacobian() const {
  assert(model);
  return model->OperatorJacobian(x, coefficients, bases);
}

Eigen::MatrixXd SupportPoint::OperatorJacobian(Eigen::VectorXd const& loc) const {
  assert(model);
  assert(loc.size()==model->inputDimension);
  return model->OperatorJacobian(loc, coefficients, bases);
}

Eigen::MatrixXd SupportPoint::OperatorJacobian(Eigen::VectorXd const& loc, Eigen::VectorXd const& coeffs) const {
  assert(model);
  assert(loc.size()==model->inputDimension);
  assert(coeffs.size()==numCoefficients);
  return model->OperatorJacobian(loc, coeffs, bases);
}

std::vector<Eigen::MatrixXd> SupportPoint::OperatorHessian(Eigen::VectorXd const& loc, Eigen::VectorXd const& coefficients) const {
  assert(loc.size()==model->inputDimension);
  assert(coefficients.size()==numCoefficients);
  return model->OperatorHessian(loc, coefficients, bases);
}

double SupportPoint::MinimizeUncoupledCost(Eigen::MatrixXd const& forcing, pt::ptree const& options) {
  assert(uncoupledCost);

  // set the forcing in the uncoupled cost 
  uncoupledCost->SetForcingEvaluations(forcing);

  // compute the optimial parameters 
  double cst = MinimizeUncoupledCost(options);

  // unset the forcing 
  uncoupledCost->UnsetForcingEvaluations();

  return cst;
}

double SupportPoint::MinimizeUncoupledCost(pt::ptree const& options) {
  assert(uncoupledCost);

  // if linear, use the quadratic cost
  if( quadOptimizer || uncoupledCost->IsQuadratic() ) {
    // have we already computed the matrix decomposition?
    if( !quadOptimizer ) { quadOptimizer = std::make_shared<DenseQuadraticCostOptimizer>(uncoupledCost, options); }
    quadOptimizer->Minimize(coefficients);
    return 0.0;
  }

  auto opt = Optimizer<Eigen::MatrixXd>::Construct(uncoupledCost, options);
  const std::pair<Optimization::Convergence, double> info = opt->Minimize(coefficients);
  assert(info.first>0);
  return info.second;
}

std::vector<Eigen::VectorXd> SupportPoint::EvaluateBasisFunctions(Eigen::VectorXd const& loc) const {
  std::vector<Eigen::VectorXd> basisEvals(model->outputDimension);
  for( std::size_t i=0; i<model->outputDimension; ++i ) { basisEvals[i] = bases[i]->EvaluateBasisFunctions(loc); }
  return basisEvals;
}

Eigen::VectorXd SupportPoint::EvaluateLocalFunction(Eigen::VectorXd const& coeffs, std::vector<Eigen::VectorXd> const& basisEvals) const {
  assert(coeffs.size()==numCoefficients);
  Eigen::VectorXd output(model->outputDimension);
  std::size_t ind = 0;
  for( std::size_t i=0; i<model->outputDimension; ++i ) {
    const std::size_t basisSize = bases[i]->NumBasisFunctions();
    assert(basisEvals[i].size()==basisSize);
    output(i) = coeffs.segment(ind, basisSize).dot(basisEvals[i]);
    ind += basisSize;
  }

  return output;
}

Eigen::VectorXd SupportPoint::EvaluateLocalFunction(Eigen::VectorXd const& loc, Eigen::VectorXd const& coeffs) const {
  assert(coeffs.size()==numCoefficients);
  assert(loc.size()==model->inputDimension);
  Eigen::VectorXd output(model->outputDimension);
  std::size_t ind = 0;
  for( std::size_t i=0; i<model->outputDimension; ++i ) {
    const std::size_t basisSize = bases[i]->NumBasisFunctions();
    output(i) = coeffs.segment(ind, basisSize).dot(bases[i]->EvaluateBasisFunctions(loc));
    ind += basisSize;
  }

  return output;
}

Eigen::VectorXd SupportPoint::EvaluateLocalFunction(Eigen::VectorXd const& loc) const { return EvaluateLocalFunction(loc, coefficients); }

const std::vector<std::shared_ptr<const BasisFunctions> >& SupportPoint::GetBasisFunctions() const { return bases; }

std::size_t SupportPoint::NumNeighbors() const { return numNeighbors; }

Eigen::VectorXd SupportPoint::Coefficients() const { return coefficients; }

Eigen::VectorXd& SupportPoint::Coefficients() { return coefficients; }

double SupportPoint::ComputeUncoupledCost() const {
  assert(false);
  assert(uncoupledCost);
  //assert(coefficients.size()==uncoupledCost->inputSizes(0));
  //return uncoupledCost->Cost(coefficients);
  return 0.0;
}

double SupportPoint::ComputeCoupledCost() const {
  assert(false);
  double cost = 0.0;
  /*for( const auto& coupled : coupledCost ) {
    assert(coupled);
    auto neigh = coupled->neighbor.lock();
    assert(neigh);

    cost += coupled->Cost(coefficients, neigh->Coefficients());
  }*/
  return cost;
}

bool SupportPoint::Coupled() const { return coupledCost.size()>0; }

void SupportPoint::ComputeOptimalCoefficients(Eigen::MatrixXd const& data) {
  assert(data.cols()==NumNeighbors());
  assert(data.rows()==model->outputDimension);

  coefficients = lsJacobian*Eigen::Map<const Eigen::VectorXd>(data.data(), data.size());
  assert(coefficients.size()==numCoefficients);
}
