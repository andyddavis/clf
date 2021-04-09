#include "clf/SupportPoint.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace clf;

SupportPoint::SupportPoint(Eigen::VectorXd const& x, pt::ptree const& pt) :
ModPiece(Eigen::VectorXi::Constant(1, x.size()), Eigen::VectorXi::Constant(1, pt.get<std::size_t>("OutputDimension", 1))),
x(x) {}

std::size_t SupportPoint::InputDimension() const { return inputSizes(0); }

std::size_t SupportPoint::OutputDimension() const { return outputSizes(0); }

void SupportPoint::EvaluateImpl(ref_vector<Eigen::VectorXd> const& input) {
  assert(false);
}
