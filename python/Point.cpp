#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/eigen.h>

#include "clf/Point.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::PointWrapper(py::module& mod) {
  py::class_<Point, std::shared_ptr<Point> > point(mod, "Point");
  point.def(py::init<Eigen::VectorXd const&>());

  point.def_readonly("x", &Point::x);
}
