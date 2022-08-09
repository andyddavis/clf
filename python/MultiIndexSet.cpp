#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/stl.h>

#include "clf/MultiIndexSet.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::MultiIndexSetWrapper(py::module& mod) {
  py::class_<MultiIndexSet, std::shared_ptr<MultiIndexSet> > set(mod, "MultiIndexSet");
  
  set.def(py::init(static_cast<std::unique_ptr<MultiIndexSet> (*)(std::shared_ptr<Parameters> const&)>(&clf::MultiIndexSet::CreateTotalOrder)));
  set.def(py::init(static_cast<std::unique_ptr<MultiIndexSet> (*)(std::size_t const, std::size_t const)>(&clf::MultiIndexSet::CreateTotalOrder)));
  set.def("Dimension", &MultiIndexSet::Dimension);
  set.def("NumIndices", &MultiIndexSet::NumIndices);
  set.def("MaxIndex", &MultiIndexSet::MaxIndex);
  set.def_readonly("indices", &MultiIndexSet::indices);
}
