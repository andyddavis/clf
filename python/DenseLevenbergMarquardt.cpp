#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "clf/DenseLevenbergMarquardt.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::DenseLevenbergMarquardtWrapper(pybind11::module& mod) {
  py::class_<DenseLevenbergMarquardt, std::shared_ptr<DenseLevenbergMarquardt>, LevenbergMarquardt<Eigen::MatrixXd> > dense(mod, "DenseLevenbergMarquardt");
  dense.def(py::init<std::shared_ptr<const DenseCostFunction> const&, std::shared_ptr<Parameters> const&>());
}
