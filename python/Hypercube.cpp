#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "clf/Hypercube.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::HypercubeWrapper(py::module& mod) {
  py::class_<Hypercube, std::shared_ptr<Hypercube>, Domain> cube(mod, "Hypercube");
  cube.def(py::init<std::size_t const>());
  cube.def(py::init<std::size_t const, std::shared_ptr<const Parameters> const&>());
  cube.def(py::init<double const, double const, std::size_t const>());
  cube.def(py::init<double const, double const, std::size_t const, std::shared_ptr<const Parameters> const&>());
  cube.def(py::init<std::vector<bool> const&, double const, double const>());
  cube.def(py::init<std::vector<bool> const&, double const, double const, std::shared_ptr<const Parameters> const&>());
  cube.def(py::init<std::shared_ptr<Parameters> const&>());
  cube.def(py::init<std::vector<bool> const&>());
  cube.def(py::init<std::vector<bool> const&, std::shared_ptr<Parameters> const&>());
  cube.def(py::init<Eigen::VectorXd const&, Eigen::VectorXd const&>());
  cube.def(py::init<Eigen::VectorXd const&, Eigen::VectorXd const&, std::shared_ptr<const Parameters> const&>());
  cube.def(py::init<Eigen::VectorXd const&, Eigen::VectorXd const&, std::vector<bool> const&>());
  cube.def(py::init<Eigen::VectorXd const&, Eigen::VectorXd const&, std::vector<bool> const&, std::shared_ptr<const Parameters> const&>());

  cube.def("LeftBoundary", &Hypercube::LeftBoundary);
  cube.def("RightBoundary", &Hypercube::RightBoundary);
  cube.def("MapPeriodic", &Hypercube::MapPeriodic);
}
