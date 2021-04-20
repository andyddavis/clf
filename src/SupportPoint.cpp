#include "clf/SupportPoint.hpp"

#include "clf/UtilityFunctions.hpp"
#include "clf/PolynomialBasis.hpp"
#include "clf/SinCosBasis.hpp"
#include "clf/SupportPointCloud.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace clf;

SupportPoint::SupportPoint(Eigen::VectorXd const& x, std::shared_ptr<const Model> const& model, pt::ptree const& pt) :
x(x),
model(model),
bases(CreateBasisFunctions(model->inputDimension, model->outputDimension, pt)),
numNeighbors(DetermineNumNeighbors(bases, pt)),
delta(pt.get<double>("InitialRadius", 1.0))
{
  assert(bases.size()==model->outputDimension);
  for( const auto& it : bases ) { assert(it); }
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

std::vector<std::shared_ptr<const BasisFunctions> > SupportPoint::CreateBasisFunctions(std::size_t const indim, std::size_t const outdim, pt::ptree pt) {
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
    bases.push_back(CreateBasisFunctions(indim, pt.get_child(basisOptionName)));

    // remove this basis from the list of options
    basisOptionNames.erase(0, pos==std::string::npos? pos : pos+1);
  }

  return bases;
}

std::shared_ptr<const BasisFunctions> SupportPoint::CreateBasisFunctions(std::size_t const indim, pt::ptree pt) {
  // find the time we are trying to create and make sure it is a valid option
  const std::string type = UtilityFunctions::ToUpper(pt.get<std::string>("Type"));
  if( std::find(exceptions::SupportPointInvalidBasisException::options.begin(), exceptions::SupportPointInvalidBasisException::options.end(), type)==exceptions::SupportPointInvalidBasisException::options.end() ) { throw exceptions::SupportPointInvalidBasisException(type); }

  // create the basis and return it
  if( type=="TOTALORDERPOLYNOMIALS" ) {
    pt.put("InputDimension", indim);
    return PolynomialBasis::TotalOrderBasis(pt);
  } else if( type=="TOTALORDERSINCOS" ) {
    pt.put("InputDimension", indim);
    return SinCosBasis::TotalOrderBasis(pt);
  }

  // invalid basis type, throw and exception
  throw exceptions::SupportPointInvalidBasisException(type);
  return nullptr;
}

double SupportPoint::Radius() const { return delta; }

double& SupportPoint::Radius() { return delta; }

Eigen::VectorXd SupportPoint::LocalCoordinate(Eigen::VectorXd const& y) const {
  assert(y.size()==x.size());
  return (y-x)/delta;
}

Eigen::VectorXd SupportPoint::GlobalCoordinate(Eigen::VectorXd const& xhat) const {
  assert(xhat.size()==x.size());
  return delta*xhat + x;
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

  squaredNeighborDistances.resize(indices.size());
  globalNeighorIndices.resize(indices.size());
  for( std::size_t i=0; i<indices.size(); ++i ) {
    squaredNeighborDistances[i] = neighDist[indices[i]];
    globalNeighorIndices[i] = neighInd[indices[i]];
  }

  // reset the scaling parameter for the squared neighbor distance
  delta = std::sqrt(*(squaredNeighborDistances.end()-1));
}

std::vector<std::size_t> SupportPoint::GlobalNeighborIndices() const { return globalNeighorIndices; }

std::size_t SupportPoint::NumCoefficients() const {
  std::size_t num = 0;
  for( const auto& it : bases ) { num += it->NumBasisFunctions(); }
  return num;
}

Eigen::VectorXd SupportPoint::NearestNeighbor(std::size_t const jnd) const {
  // get the cloud
  auto cld = cloud.lock();
  assert(cld);

  assert(jnd<globalNeighorIndices.size());
  return cld->GetSupportPoint(globalNeighorIndices[jnd])->x;
}

Eigen::VectorXd SupportPoint::NearestNeighborKernel() const {
  Eigen::VectorXd kernel(squaredNeighborDistances.size());
  const double delta2 = delta*delta;
  for( std::size_t i=0; i<kernel.size(); ++i ) { kernel(i) = model->NearestNeighborKernel(squaredNeighborDistances[i]/delta2); }
  return kernel;
}

Eigen::VectorXd SupportPoint::Operator(Eigen::VectorXd const& loc, Eigen::VectorXd const& coefficients) const {
  assert(loc.size()==model->inputDimension);
  assert(coefficients.size()==NumCoefficients());
  return model->Operator(LocalCoordinate(loc), coefficients, bases);
}

void SupportPoint::MinimizeUncoupledCost() {
  std::cout << "Minimize uncoupled cost" << std::endl;
}
