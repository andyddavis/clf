#include "clf/python/Pybind11Wrappers.hpp"

#include "clf/CoupledLocalFunctions.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::CoupledLocalFunctionsWrapper(py::module& mod) {
  py::class_<CoupledLocalFunctions, std::shared_ptr<CoupledLocalFunctions> > func(mod, "CoupledLocalFunctions");
  //func.def(py::init<std::shared_ptr<PointCloud> const&>());

  //func.def("NumLocalFunctions", &CoupledLocalFunctions::NumLocalFunctions);
}
