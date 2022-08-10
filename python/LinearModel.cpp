#include "clf/python/Pybind11Wrappers.hpp"
#include "clf/python/PySystemOfEquations.hpp"

#include "clf/LinearModel.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::LinearModelWrapper(py::module& mod) {
  py::class_<LinearModel, std::shared_ptr<LinearModel>, SystemOfEquations, python::PySystemOfEquations<LinearModel> > sys(mod, "LinearModel");
  sys.def(py::init<std::size_t const, std::size_t const>());
  sys.def(py::init<std::size_t const, std::size_t const, std::shared_ptr<const Parameters> const&>());
  sys.def(py::init<std::shared_ptr<const Parameters> const&>());
  sys.def(py::init<std::size_t const, std::size_t const, std::size_t const>());
  sys.def(py::init<std::size_t const, std::size_t const, std::size_t const, std::shared_ptr<const Parameters> const&>());
  sys.def(py::init<Eigen::MatrixXd const&, std::shared_ptr<const Parameters> const&>());
  sys.def(py::init<std::size_t const, Eigen::MatrixXd const&>());
  sys.def(py::init<std::size_t const, Eigen::MatrixXd const&, std::shared_ptr<const Parameters> const&>());

  sys.def("Operator", static_cast<Eigen::MatrixXd(LinearModel::*)(Eigen::VectorXd const&) const>(&LinearModel::Operator));
  sys.def("Operator", static_cast<Eigen::VectorXd(LinearModel::*)(std::shared_ptr<LocalFunction> const&, Eigen::VectorXd const&, Eigen::VectorXd const&) const>(&LinearModel::Operator));
}
