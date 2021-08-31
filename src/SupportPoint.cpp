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

SupportPoint::SupportPoint(Eigen::VectorXd const& x, std::shared_ptr<const Model> const& model, pt::ptree const& pt) :
x(x),
model(model),
optimizationOptions(GetOptimizationOptions(pt))
{}

pt::ptree SupportPoint::GetOptimizationOptions(pt::ptree const& pt) {
  auto opt = pt.get_child_optional("Optimization");
  if( !opt ) { return pt::ptree(); }
  return opt.get();
}

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

void SupportPoint::SetNearestNeighbors(std::shared_ptr<const SupportPointCloud> const& newcloud, std::vector<std::size_t> const& neighInd, std::vector<double> const& neighDist) {
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

std::vector<std::size_t> SupportPoint::GlobalNeighborIndices() const { return globalNeighorIndices; }

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

void SupportPoint::ComputeNearestNeighborKernel() {
  nearestNeighborKernel.resize(squaredNeighborDistances.size());

  for( std::size_t i=0; i<nearestNeighborKernel.size(); ++i ) { nearestNeighborKernel(i) = model->NearestNeighborKernel(squaredNeighborDistances[i]/(*(squaredNeighborDistances.end()-1))); }
}

Eigen::VectorXd SupportPoint::NearestNeighborKernel() const { return nearestNeighborKernel; }

double SupportPoint::NearestNeighborKernel(std::size_t const ind) const {
  assert(ind<nearestNeighborKernel.size());
  return nearestNeighborKernel(ind);
}

Eigen::VectorXd SupportPoint::Operator(Eigen::VectorXd const& loc, Eigen::VectorXd const& coefficients) const {
  assert(loc.size()==model->inputDimension);
  assert(coefficients.size()==numCoefficients);
  return model->Operator(loc, coefficients, bases);
}

Eigen::MatrixXd SupportPoint::OperatorJacobian(Eigen::VectorXd const& loc, Eigen::VectorXd const& coefficients) const {
  assert(loc.size()==model->inputDimension);
  assert(coefficients.size()==numCoefficients);
  return model->OperatorJacobian(loc, coefficients, bases);
}

std::vector<Eigen::MatrixXd> SupportPoint::OperatorHessian(Eigen::VectorXd const& loc, Eigen::VectorXd const& coefficients) const {
  assert(loc.size()==model->inputDimension);
  assert(coefficients.size()==numCoefficients);
  return model->OperatorHessian(loc, coefficients, bases);
}

double SupportPoint::MinimizeUncoupledCostNLOPT() {
  /*pt::ptree options;
  options.put("Ftol.AbsoluteTolerance", optimizationOptions.atol_function);
  options.put("Ftol.RelativeTolerance", optimizationOptions.rtol_function);
  options.put("Xtol.AbsoluteTolerance", optimizationOptions.atol_step);
  options.put("Xtol.RelativeTolerance", optimizationOptions.rtol_step);
  options.put("MaxEvaluations", optimizationOptions.maxEvals);
  options.put("Algorithm", optimizationOptions.algNLOPT);

  auto opt = std::make_shared<NLoptOptimizer>(uncoupledCost, options);

  double costVal;
  std::tie(coefficients, costVal) = opt->Solve(std::vector<Eigen::VectorXd>(1, coefficients));

  return costVal;*/
  return 0.0;
}

std::pair<double, double> SupportPoint::LineSearch(Eigen::VectorXd const& coefficients, Eigen::VectorXd const& stepDir, double const prevCost) const {
  double alpha = optimizationOptions.maxStepSizeScale;
  //double newCost = uncoupledCost->Cost((coefficients-alpha*stepDir).eval());
  double newCost = 0.0;
  std::size_t lineSearchIter = 0;
  while( newCost>prevCost || lineSearchIter++>optimizationOptions.maxLineSearchIterations ) {
    alpha *= optimizationOptions.lineSearchFactor;
    //newCost = uncoupledCost->Cost((coefficients-alpha*stepDir).eval());
  }

  return std::pair<double, double>(alpha, newCost);
}

Eigen::VectorXd SupportPoint::StepDirection(Eigen::VectorXd const& coefficients, Eigen::VectorXd const& grad,  bool const useGN) const {
  // compute the Hessian
  const Eigen::MatrixXd hess(grad.size(), grad.size());
  //const Eigen::MatrixXd hess = uncoupledCost->Hessian(coefficients, useGN);
  //assert(hess.rows()==coefficients.size()); assert(hess.cols()==coefficients.size());

  // step direction xk-xkp1 = H^{-1} g
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver;
  solver.compute(hess);
  return solver.solve(grad);
}

double SupportPoint::MinimizeUncoupledCostNewton() {
  //const double initCost = uncoupledCost->Cost(coefficients);
  const double initCost = 1.0;
  double prevCost = initCost;
  for( std::size_t iter=0; iter<optimizationOptions.maxEvals; ++iter ) {
    // compute the cost function gradient
    //const Eigen::VectorXd grad = uncoupledCost->Gradient(coefficients);
    const Eigen::VectorXd grad(10);

    // if the gradient is small, break
    if( grad.norm()<optimizationOptions.atol_grad ) { break; }

    // compute the step direction
    Eigen::VectorXd stepDir = StepDirection(coefficients, grad, optimizationOptions.useGaussNewtonHessian);

    // do the line search
    double alpha, newCost;
    std::tie(alpha, newCost) = LineSearch(coefficients, stepDir, prevCost);

    // if we are not using the Gauss-Newton Hessian and the stepsize is small, try taking a step with the Gauss-Newton Hessian
    if( !optimizationOptions.useGaussNewtonHessian && alpha<optimizationOptions.atol_step ) {
      // compute a new step direction
      stepDir = StepDirection(coefficients, grad, true);

      // do the line search
      std::tie(alpha, newCost) = LineSearch(coefficients, stepDir, prevCost);
    }

    // make a step
    if( alpha<1.0-1.0e-10 ) { stepDir *= alpha; }
    const double stepsize = stepDir.norm();
    prevCost = newCost;
    coefficients -= stepDir;

    // check convergence
    if(
      prevCost<optimizationOptions.atol_function |
      prevCost/std::max(1.0e-10, initCost)<optimizationOptions.rtol_function |
      stepsize<optimizationOptions.atol_step |
      stepsize/std::max(1.0e-10, coefficients.norm())<optimizationOptions.rtol_step
    ) { break; }
  }

  return prevCost;
}

double SupportPoint::MinimizeUncoupledCost() {
  assert(uncoupledCost);

  pt::ptree pt;
  auto lm = std::make_shared<SparseLevenbergMarquardt>(uncoupledCost, pt);

  if( coefficients.size()!=uncoupledCost->inDim ) { coefficients = Eigen::VectorXd::Zero(uncoupledCost->inDim); }
  Eigen::VectorXd cost;
  lm->Minimize(coefficients, cost);

  return cost.dot(cost);
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
  assert(uncoupledCost);
  //assert(coefficients.size()==uncoupledCost->inputSizes(0));
  //return uncoupledCost->Cost(coefficients);
  return 0.0;
}

double SupportPoint::ComputeCoupledCost() const {
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
