#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "clf/LevenbergMarquardt.hpp"

namespace py = pybind11;

namespace clf {
namespace python {

/// Add the clf::CostFunction base to the python interface
/**
   @param[in] mod The python module
   @param[in] name The name of the base class
*/
template<typename MatrixType>
void LevenbergMarquardtBaseWrapper(py::module& mod, std::string const& name) {
  py::class_<LevenbergMarquardt<MatrixType>, std::shared_ptr<LevenbergMarquardt<MatrixType> > > lm(mod, name.c_str());

  lm.def("NumParameters", &LevenbergMarquardt<MatrixType>::NumParameters);
  lm.def("Minimize", static_cast<std::tuple<Optimization::Convergence, double, Eigen::VectorXd, Eigen::VectorXd> (LevenbergMarquardt<MatrixType>::*)(Eigen::VectorXd const&)>(&LevenbergMarquardt<MatrixType>::Minimize));
}

} // namespace python
} // namespace clf

using namespace clf;

void clf::python::LevenbergMarquardtWrapper(pybind11::module& mod) {
  py::enum_<Optimization::Convergence> conv(mod, "OptimizationConvergence");
  conv.value("FAILED_MAX_HESSIAN_EVALUATIONS", Optimization::Convergence::FAILED_MAX_HESSIAN_EVALUATIONS);
  conv.value("FAILED_MAX_JACOBIAN_EVALUATIONS", Optimization::Convergence::FAILED_MAX_JACOBIAN_EVALUATIONS);
  conv.value("FAILED_MAX_COST_EVALgUATIONS", Optimization::Convergence::FAILED_MAX_COST_EVALUATIONS);
  conv.value("FAILED_MAX_ITERATIONS", Optimization::Convergence::FAILED_MAX_ITERATIONS);
  conv.value("FAILED", Optimization::Convergence::FAILED);
  conv.value("CONTINUE_RUNNING", Optimization::Convergence::CONTINUE_RUNNING);
  conv.value("CONVERGED", Optimization::Convergence::CONVERGED);
  conv.value("CONVERGED_FUNCTION_SMALL", Optimization::Convergence::CONVERGED_FUNCTION_SMALL);
  conv.value("CONVERGED_GRADIENT_SMALL", Optimization::Convergence::CONVERGED_GRADIENT_SMALL);
  
  python::LevenbergMarquardtBaseWrapper<Eigen::MatrixXd>(mod, "DenseLevenbergMarquardtBase");
  python::LevenbergMarquardtBaseWrapper<Eigen::SparseMatrix<double> >(mod, "SparseLevenbergMarquardtBase");
}
