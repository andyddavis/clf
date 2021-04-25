#include <pybind11/pybind11.h>

#include <MUQ/Utilities/PyDictConversion.h>

#include "clf/Pybind11Wrappers.hpp"

#include "clf/LocalFunction.hpp"

namespace py = pybind11;
using namespace muq::Utilities;
using namespace clf;

void clf::python::LocalFunctionWrapper(pybind11::module& mod) {
  py::class_<LocalFunction, std::shared_ptr<LocalFunction> > func(mod, "LocalFunction");
  func.def(py::init( [] (std::shared_ptr<SupportPointCloud> const& cloud, py::dict const& d) { return new LocalFunction(cloud, ConvertDictToPtree(d)); }));
  func.def("CoefficientCost", &LocalFunction::CoefficientCost);
  func.def("Evaluate", &LocalFunction::Evaluate);
  func.def("NearestNeighbor", &LocalFunction::NearestNeighbor);
  func.def("NearestNeighborIndex", &LocalFunction::NearestNeighborIndex);
  func.def("NearestNeighborDistance", &LocalFunction::NearestNeighborDistance);
}
