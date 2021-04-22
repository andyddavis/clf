#include "clf/SupportPointBasis.hpp"

#include "clf/SupportPoint.hpp"

using namespace clf;

SupportPointBasis::SupportPointBasis(std::shared_ptr<SupportPoint> const& point, std::shared_ptr<const BasisFunctions> const& basis, boost::property_tree::ptree const& pt) : BasisFunctions(basis->multis, pt), basis(basis), point(point), delta(pt.get<double>("Radius", 1.0)) {}

double SupportPointBasis::ScalarBasisFunction(double const x, std::size_t const ind, std::size_t const coordinate) const { return basis->ScalarBasisFunction(LocalCoordinate(x, coordinate), ind, coordinate); }

double SupportPointBasis::ScalarBasisFunctionDerivative(double const x, std::size_t const ind, std::size_t const coordinate, std::size_t const k) const { return basis->ScalarBasisFunctionDerivative(LocalCoordinate(x, coordinate), ind, coordinate, k)/std::pow(delta, (double)k); }

double SupportPointBasis::Radius() const { return delta; }

void SupportPointBasis::SetRadius(double const newdelta) const { delta = newdelta; }

double SupportPointBasis::LocalCoordinate(double const xi, std::size_t const coord) const {
  auto pnt = point.lock();
  assert(pnt);
  assert(coord<pnt->x.size());

  return (xi-pnt->x(coord))/delta;
}

Eigen::VectorXd SupportPointBasis::LocalCoordinate(Eigen::VectorXd const& x) const {
  auto pnt = point.lock();
  assert(pnt);

  assert(x.size()==pnt->x.size());
  return (x-pnt->x)/delta;
}

Eigen::VectorXd SupportPointBasis::GlobalCoordinate(Eigen::VectorXd const& xhat) const {
  auto pnt = point.lock();
  assert(pnt);

  assert(xhat.size()==pnt->x.size());
  return delta*xhat + pnt->x;
}
