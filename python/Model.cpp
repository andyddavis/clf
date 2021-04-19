#include <pybind11/pybind11.h>

#include <MUQ/Utilities/PyDictConversion.h>

#include "clf/Pybind11Wrappers.hpp"

#include "clf/Model.hpp"

namespace py = pybind11;
using namespace muq::Utilities;
using namespace clf;

void clf::python::ModelWrapper(pybind11::module& mod) {
  py::class_<Model, std::shared_ptr<Model> > model(mod, "Model");
  model.def(py::init( [] (py::dict const& d) { return new Model(ConvertDictToPtree(d)); }));
  model.def_readonly("inputDimension", &Model::inputDimension);
  model.def_readonly("outputDimension", &Model::outputDimension);
}
