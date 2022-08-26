#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "clf/CostFunction.hpp"

namespace py = pybind11;

namespace clf {
namespace python {

/// Add the clf::CostFunction base to the python interface
/**
   @param[in] mod The python module
   @param[in] name The name of the base class
*/
template<typename MatrixType>
void CostFunctionBaseWrapper(py::module& mod, std::string const& name) {
  py::class_<CostFunction<MatrixType>, std::shared_ptr<CostFunction<MatrixType> > > func(mod, name.c_str());

  func.def("InputDimension", &CostFunction<MatrixType>::InputDimension);
  func.def("Evaluate", &CostFunction<MatrixType>::Evaluate);
  func.def("Jacobian", &CostFunction<MatrixType>::Jacobian);
  func.def("Gradient", static_cast<Eigen::VectorXd(CostFunction<MatrixType>::*)(Eigen::VectorXd const&) const>(&CostFunction<MatrixType>::Gradient));
  func.def("Gradient", static_cast<Eigen::VectorXd(CostFunction<MatrixType>::*)(Eigen::VectorXd const&, MatrixType const& jac) const>(&CostFunction<MatrixType>::Gradient));
  func.def("Hessian", [](CostFunction<MatrixType>& self, Eigen::VectorXd const& beta) { return self.Hessian(beta); } );
  func.def("Hessian", static_cast<MatrixType(CostFunction<MatrixType>::*)(Eigen::VectorXd const&, bool const) const>(&CostFunction<MatrixType>::Hessian));
  func.def("Hessian", [](CostFunction<MatrixType>& self, Eigen::VectorXd const& beta, MatrixType const& jac) { return self.Hessian(beta, jac); } );
  func.def("Hessian", static_cast<MatrixType(CostFunction<MatrixType>::*)(Eigen::VectorXd const&, Eigen::VectorXd const&, MatrixType const&, bool const) const>(&CostFunction<MatrixType>::Hessian));
  func.def("Hessian", [](CostFunction<MatrixType>& self, Eigen::VectorXd const& beta, Eigen::MatrixXd const& cost, MatrixType const& jac) { return self.Hessian(beta, cost, jac); } );
  func.def("Hessian", static_cast<MatrixType(CostFunction<MatrixType>::*)(Eigen::VectorXd const&, MatrixType const&, bool const) const>(&CostFunction<MatrixType>::Hessian));
  func.def_readonly("numPenaltyFunctions", &CostFunction<MatrixType>::numPenaltyFunctions);
  func.def_readonly("numTerms", &CostFunction<MatrixType>::numTerms);
}

} // namespace python
} // namespace clf

void clf::python::CostFunctionWrapper(pybind11::module& mod) {
  clf::python::CostFunctionBaseWrapper<Eigen::MatrixXd>(mod, "DenseCostFunctionBase");
  clf::python::CostFunctionBaseWrapper<Eigen::SparseMatrix<double> >(mod, "SparseCostFunctionBase");
}
