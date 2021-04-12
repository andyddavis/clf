#include "clf/SupportPoint.hpp"

#include "clf/UtilityFunctions.hpp"
#include "clf/PolynomialBasis.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace clf;

SupportPoint::SupportPoint(Eigen::VectorXd const& x, pt::ptree const& pt) :
ModPiece(Eigen::VectorXi::Constant(1, x.size()), Eigen::VectorXi::Constant(1, pt.get<std::size_t>("OutputDimension", 1))),
x(x),
basis(CreateBasisFunctions(x.size(), pt.get_child("BasisFunctions"))),
delta(pt.get<double>("InitialRadius", 1.0))
{
  assert(basis);
}

std::shared_ptr<BasisFunctions> SupportPoint::CreateBasisFunctions(std::size_t const indim, pt::ptree pt) {
  const std::string type = UtilityFunctions::ToUpper(pt.get<std::string>("Type"));

  if( type=="TOTALORDERPOLYNOMIALS" ) {
    pt.put("InputDimension", indim);
    return PolynomialBasis::TotalOrderPolynomialBasis(pt);
  }

  // invalid basis type, throw and exception
  throw SupportPointBasisException(type);
  return nullptr;
}

std::size_t SupportPoint::InputDimension() const { return inputSizes(0); }

std::size_t SupportPoint::OutputDimension() const { return outputSizes(0); }

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

void SupportPoint::EvaluateImpl(ref_vector<Eigen::VectorXd> const& input) {

  assert(false);
}
