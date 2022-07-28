#include "clf/python/Pybind11Wrappers.hpp"

namespace py = pybind11;
              
PYBIND11_MODULE(PyCoupledLocalFunctions, module) {
  clf::python::ParametersWrapper(module);
}
