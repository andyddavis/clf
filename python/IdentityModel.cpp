#include "clf/python/Pybind11Wrappers.hpp"
#include "clf/python/PySystemOfEquations.hpp"

#include "clf/IdentityModel.hpp"

namespace py = pybind11;
using namespace clf;  

void clf::python::IdentityModelWrapper(py::module& mod) {
  py::class_<IdentityModel, std::shared_ptr<IdentityModel>, SystemOfEquations, python::PySystemOfEquations<IdentityModel> > sys(mod, "IdentityModel");
  sys.def(py::init<std::size_t const, std::size_t const>());
  sys.def(py::init<std::size_t const, std::size_t const, std::shared_ptr<const Parameters> const&>());
  sys.def(py::init<std::shared_ptr<const Parameters> const&>());
}
