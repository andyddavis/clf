#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/stl.h>

#include "clf/Parameters.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::ParametersWrapper(py::module& mod) {
  py::class_<Parameters, std::shared_ptr<Parameters> > para(mod, "Parameters");
  para.def(py::init<>());

  para.def("NumParameters", &Parameters::NumParameters);
  para.def("Add", &Parameters::Add);
  para.def("Get", static_cast<Parameters::Parameter (Parameters::*)(std::string const&) const>(&Parameters::Get<Parameters::Parameter>));
  para.def("Get", static_cast<Parameters::Parameter (Parameters::*)(std::string const&, Parameters::Parameter const&) const>(&Parameters::Get<Parameters::Parameter>));
}
