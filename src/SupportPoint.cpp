#include "clf/SupportPoint.hpp"

#include "clf/UtilityFunctions.hpp"
#include "clf/PolynomialBasis.hpp"
#include "clf/SinCosBasis.hpp"

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

std::vector<std::size_t> SupportPoint::DetermineNumNeighbors(std::vector<std::shared_ptr<const BasisFunctions> > const& bases, pt::ptree const& pt) {
  const std::size_t defaultNumNeighs = pt.get<std::size_t>("NumNeighbors", std::numeric_limits<std::size_t>::max());

  std::vector<std::size_t> numNeighs(bases.size());
  for( std::size_t d=0; d<bases.size(); ++d ) {
    numNeighs[d] = pt.get<std::size_t>("NumNeighbors-"+std::to_string(d),  (defaultNumNeighs<std::numeric_limits<std::size_t>::max()? defaultNumNeighs : bases[d]->NumBasisFunctions()+1));
    if( numNeighs[d]<bases[d]->NumBasisFunctions() ) { throw exceptions::SupportPointWrongNumberOfNearestNeighbors(d, bases[d]->NumBasisFunctions(), numNeighs[d]); }
  }

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

void SupportPoint::SetNearestNeighbors(std::vector<std::size_t> const& neighInd, std::vector<double> const& neighDist) {
  assert(neighInd.size()==neighDist.size());
  assert(neighInd.size()==*std::max_element(numNeighbors.begin(), numNeighbors.end()));

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
}

std::vector<std::size_t> SupportPoint::GlobalNeighborIndices(std::size_t const outdim) const {
  assert(outdim<numNeighbors.size());
  assert(numNeighbors[outdim]<=globalNeighorIndices.size());
  return std::vector<std::size_t>(globalNeighorIndices.begin(), globalNeighorIndices.begin()+numNeighbors[outdim]);
}
