
#include "clf/Pybind11Wrappers.hpp"

namespace py = pybind11;

PYBIND11_MODULE(CoupledLocalFunctions, module) {
  clf::python::SupportPointWrapper(module);
}
