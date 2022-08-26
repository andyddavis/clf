#include "clf/python/Pybind11Wrappers.hpp"
#include "clf/python/PySystemOfEquations.hpp"

#include "clf/SystemOfEquations.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::SystemOfEquationsWrapper(py::module& mod) {
  py::class_<SystemOfEquations, std::shared_ptr<SystemOfEquations>, python::PySystemOfEquations<SystemOfEquations> > sys(mod, "SystemOfEquations");
  sys.def(py::init<std::size_t const, std::size_t const>());
  sys.def(py::init<std::size_t const, std::size_t const, std::shared_ptr<const Parameters> const&>());
  sys.def(py::init<std::shared_ptr<const Parameters> const&>());

  sys.def("RightHandSide", &SystemOfEquations::RightHandSide);
  sys.def("Operator", &SystemOfEquations::Operator);
  sys.def("JacobianWRTCoefficients", &SystemOfEquations::JacobianWRTCoefficients);
  sys.def("JacobianWRTCoefficientsFD", &SystemOfEquations::JacobianWRTCoefficientsFD);
  sys.def("HessianWRTCoefficients", &SystemOfEquations::HessianWRTCoefficients);
  sys.def("HessianWRTCoefficientsFD", &SystemOfEquations::HessianWRTCoefficientsFD);
  sys.def_readonly("indim", &SystemOfEquations::indim);
  sys.def_readonly("outdim", &SystemOfEquations::outdim);
  sys.def_readonly("id", &SystemOfEquations::id);
}
