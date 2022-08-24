#include "clf/python/Pybind11Wrappers.hpp"
#include "clf/python/PySystemOfEquations.hpp"

#include <pybind11/eigen.h>

#include "clf/BurgersEquation.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::BurgersEquationWrapper(py::module& mod) {
  py::class_<BurgersEquation, std::shared_ptr<BurgersEquation>, ConservationLaw, SystemOfEquations, python::PySystemOfEquations<BurgersEquation> > sys(mod, "BurgersEquation");
  sys.def(py::init<std::size_t const>());
  sys.def(py::init<std::size_t const, double const>());
  sys.def(py::init<std::size_t const, double const, std::shared_ptr<const Parameters> const&>());
  sys.def(py::init<Eigen::VectorXd const&>());
  sys.def(py::init<Eigen::VectorXd const&, std::shared_ptr<const Parameters> const&>());
}
