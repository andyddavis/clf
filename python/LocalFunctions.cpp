#include <pybind11/pybind11.h>

#include <MUQ/Utilities/PyDictConversion.h>

#include "clf/Pybind11Wrappers.hpp"

#include "clf/LocalFunctions.hpp"

namespace py = pybind11;
using namespace muq::Utilities;
using namespace clf;

void clf::python::LocalFunctionsWrapper(pybind11::module& mod) {
  py::class_<LocalFunctions, std::shared_ptr<LocalFunctions> > func(mod, "LocalFunctions");
  func.def(py::init( [] (std::shared_ptr<SupportPointCloud> const& cloud, py::dict const& d) { return new LocalFunctions(cloud, ConvertDictToPtree(d)); }));
  func.def("CoefficientCost", &LocalFunctions::CoefficientCost);
  func.def("Evaluate", &LocalFunctions::Evaluate);
  func.def("NearestNeighbor", &LocalFunctions::NearestNeighbor);
  func.def("NearestNeighborIndex", &LocalFunctions::NearestNeighborIndex);
  func.def("NearestNeighborDistance", &LocalFunctions::NearestNeighborDistance);
}
