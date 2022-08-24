#include "clf/python/Pybind11Wrappers.hpp"
#include "clf/python/PySystemOfEquations.hpp"

#include <pybind11/eigen.h>

#include "clf/AdvectionEquation.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::AdvectionEquationWrapper(py::module& mod) {
  py::class_<AdvectionEquation, std::shared_ptr<AdvectionEquation>, ConservationLaw, SystemOfEquations, python::PySystemOfEquations<AdvectionEquation> > sys(mod, "AdvectionEquation");
  sys.def(py::init<std::size_t const>());
  sys.def(py::init<std::size_t const, double const>());
  sys.def(py::init<std::size_t const, double const, std::shared_ptr<const Parameters> const&>());
  sys.def(py::init<Eigen::VectorXd const&>());
  sys.def(py::init<Eigen::VectorXd const&, std::shared_ptr<const Parameters> const&>());
}
