#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/stl.h>

#include "clf/MultiIndex.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::MultiIndexWrapper(py::module& mod) {
  py::class_<MultiIndex, std::shared_ptr<MultiIndex> > ind(mod, "MultiIndex");
  
  ind.def(py::init<std::vector<std::size_t> const&>());
  ind.def("Dimension", &MultiIndex::Dimension);
  ind.def("Order", &MultiIndex::Order);
  ind.def_readonly("alpha", &MultiIndex::alpha);
}

