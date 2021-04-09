#include "clf/SupportPoint.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace clf;

SupportPoint::SupportPoint(Eigen::VectorXd const& x, pt::ptree const& pt) :
ModPiece(Eigen::VectorXi::Constant(1, x.size()), Eigen::VectorXi::Constant(1, pt.get<std::size_t>("OutputDimension", 1))),
delta(pt.get<double>("InitialRadius", 1.0)),
x(x) {}

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
