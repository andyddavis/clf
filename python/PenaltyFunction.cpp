#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "clf/PenaltyFunction.hpp"

namespace py = pybind11;

namespace clf {
namespace python {

/// Add the clf::PenaltyFunction base to the python interface
/**
   @param[in] mod The python module
   @param[in] name The name of the base class
*/
template<typename MatrixType>
void PenaltyFunctionBaseWrapper(py::module& mod, std::string const& name) {
  py::class_<PenaltyFunction<MatrixType>, std::shared_ptr<PenaltyFunction<MatrixType> > > func(mod, name.c_str());

  func.def("JacobianFD", &PenaltyFunction<MatrixType>::JacobianFD);
  func.def("HessianFD", &PenaltyFunction<MatrixType>::HessianFD);
  func.def("InputDimension", &PenaltyFunction<MatrixType>::InputDimension);
  func.def("OutputDimension", &PenaltyFunction<MatrixType>::OutputDimension);
}

} // namespace python
} // namespace clf

void clf::python::PenaltyFunctionWrapper(py::module& mod) {    
  clf::python::PenaltyFunctionBaseWrapper<Eigen::MatrixXd>(mod, "DensePenaltyFunctionBase");
  clf::python::PenaltyFunctionBaseWrapper<Eigen::SparseMatrix<double> >(mod, "SparsePenaltyFunctionBase");
}
