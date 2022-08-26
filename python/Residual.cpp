#include "clf/python/Pybind11Wrappers.hpp"

#include "clf/Residual.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::ResidualWrapper(pybind11::module& mod) {
  py::class_<Residual, std::shared_ptr<Residual>, DensePenaltyFunction> resid(mod, "Residual");

  resid.def("NumPoints", &Residual::NumPoints);
  resid.def("GetPoint", &Residual::GetPoint);
}
