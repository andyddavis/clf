#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/eigen.h>

#include "clf/Domain.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::DomainWrapper(py::module& mod) {
  py::class_<Domain, std::shared_ptr<Domain> > dom(mod, "Domain");
  dom.def(py::init<std::size_t const>());

  dom.def("Inside", &Domain::Inside);
  dom.def("MapToHypercube", &Domain::MapToHypercube);
  dom.def("Sample", &Domain::Sample);
  dom.def("Distance", &Domain::Distance);
}
