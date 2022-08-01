#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/eigen.h>

#include "clf/BasisFunctions.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::BasisFunctionsWrapper(py::module& mod) {
  py::class_<BasisFunctions, std::shared_ptr<BasisFunctions> > func(mod, "BasisFunctions");

  func.def("Evaluate", &BasisFunctions::Evaluate);
  func.def("EvaluateAll", &BasisFunctions::EvaluateAll);
  func.def("EvaluateDerivative", &BasisFunctions::EvaluateDerivative);
  func.def("EvaluateAllDerivatives", &BasisFunctions::EvaluateAllDerivatives);
}
