
#include "clf/Pybind11Wrappers.hpp"

namespace py = pybind11;

PYBIND11_MODULE(CoupledLocalFunctions, module) {
  clf::python::BasisFunctionsWrapper(module);
  clf::python::ModelWrapper(module);
  clf::python::SupportPointWrapper(module);
  clf::python::CoupledSupportPointWrapper(module);
  clf::python::SupportPointCloudWrapper(module);
  clf::python::LocalFunctionsWrapper(module);
}
