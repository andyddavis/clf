#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "clf/DensePenaltyFunction.hpp"

namespace py = pybind11;

namespace clf {
namespace python {

/// A trampoline class for the python interface of clf::DensePenaltyFunction
class PyDensePenaltyFunction : public DensePenaltyFunction {
public:
  // inherit the constructors
  using DensePenaltyFunction::DensePenaltyFunction;

  /// The evaluate function is pure virtual  
  /**
     @param[in] beta The input parameters \f$\beta\f$
     \return The penalty function evaluation \f$c(\beta)\f$
   */
  inline virtual Eigen::VectorXd Evaluate(Eigen::VectorXd const& beta) override { PYBIND11_OVERRIDE_PURE(Eigen::VectorXd, DensePenaltyFunction, Evaluate, beta); }

  /// Compute the Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$
  /**
     @param[in] beta The input parameters \f$\beta\f$
     \return The Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$\
   */
  inline virtual Eigen::MatrixXd Jacobian(Eigen::VectorXd const& beta) override { PYBIND11_OVERRIDE(Eigen::MatrixXd, DensePenaltyFunction, Jacobian, beta); }

    /// Compute the weighted sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} w_i \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
  /**
     @param[in] beta The input parameters \f$\beta\f$
     @param[in] weights The weights for the weighted sum
     \return The weighted sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} w_i \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
   */
  inline virtual Eigen::MatrixXd Hessian(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) { PYBIND11_OVERRIDE(Eigen::MatrixXd, DensePenaltyFunction, Hessian, beta, weights); } 

private:
};

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

using namespace clf;

void clf::python::DensePenaltyFunctionWrapper(py::module& mod) {    
  py::class_<DensePenaltyFunction, PyDensePenaltyFunction, std::shared_ptr<DensePenaltyFunction>, PenaltyFunction<Eigen::MatrixXd> > dense(mod, "DensePenaltyFunction");
  dense.def(py::init<std::size_t const, std::size_t const, std::shared_ptr<const Parameters> const&>());
  dense.def(py::init<std::size_t const, std::size_t const>());

  dense.def("Evaluate", &DensePenaltyFunction::Evaluate);
  dense.def("Jacobian", &PenaltyFunction<Eigen::MatrixXd>::Jacobian);
  dense.def("Hessian", &PenaltyFunction<Eigen::MatrixXd>::Hessian);
}
