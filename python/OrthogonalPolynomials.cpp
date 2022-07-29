#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/eigen.h>

#include "clf/LegendrePolynomials.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::OrthogonalPolynomialsWrapper(py::module& mod) {
  py::class_<OrthogonalPolynomials, std::shared_ptr<OrthogonalPolynomials> > poly(mod, "OrthogonalPolynomials");

  poly.def("Evaluate", &OrthogonalPolynomials::Evaluate);
  poly.def("EvaluateAll", &OrthogonalPolynomials::EvaluateAll);
  poly.def("EvaluateDerivative", &OrthogonalPolynomials::EvaluateDerivative);
  poly.def("EvaluateAllDerivatives", &OrthogonalPolynomials::EvaluateAllDerivatives);

  py::class_<LegendrePolynomials, std::shared_ptr<LegendrePolynomials> > leg(mod, "LegendrePolynomials", poly);
  leg.def(py::init<>());
}
