#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "clf/SparseLevenbergMarquardt.hpp"

namespace py = pybind11;
using namespace clf;

void clf::python::SparseLevenbergMarquardtWrapper(pybind11::module& mod) {
  py::class_<SparseLevenbergMarquardt, std::shared_ptr<SparseLevenbergMarquardt>, LevenbergMarquardt<Eigen::SparseMatrix<double> > > sparse(mod, "SparseLevenbergMarquardt");
  sparse.def(py::init<std::shared_ptr<const SparseCostFunction> const&, std::shared_ptr<Parameters> const&>());
}
