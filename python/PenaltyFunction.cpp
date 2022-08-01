#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/eigen.h>

#include "clf/PenaltyFunction.hpp"

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

private:
};

/// A trampoline class for the python interface of clf::SparsePenaltyFunction
class PySparsePenaltyFunction : public SparsePenaltyFunction {
public:
  // inherit the constructors
  using SparsePenaltyFunction::SparsePenaltyFunction;

  /// The evaluate function is pure virtual  
  /**
     @param[in] beta The input parameters \f$\beta\f$
     \return The penalty function evaluation \f$c(\beta)\f$
   */
  inline virtual Eigen::VectorXd Evaluate(Eigen::VectorXd const& beta) override { PYBIND11_OVERRIDE_PURE(Eigen::VectorXd, SparsePenaltyFunction, Evaluate, beta); }

private:
};

template<typename MatrixType>
void BaseWrapper(py::module& mod, std::string const& name) {
  py::class_<PenaltyFunction<MatrixType>, std::shared_ptr<PenaltyFunction<MatrixType> > > func(mod, name.c_str());

  func.def("Jacobian", &PenaltyFunction<MatrixType>::Jacobian);
  func.def("JacobianFD", &PenaltyFunction<MatrixType>::JacobianFD);
  func.def("Hessian", &PenaltyFunction<MatrixType>::Hessian);
  func.def("HessianFD", &PenaltyFunction<MatrixType>::HessianFD);
  func.def_readonly("indim", &PenaltyFunction<MatrixType>::indim);
  func.def_readonly("outdim", &PenaltyFunction<MatrixType>::outdim);
}

} // namespace python
} // namespace clf

using namespace clf;

void clf::python::PenaltyFunctionWrapper(py::module& mod) {
  python::BaseWrapper<Eigen::MatrixXd>(mod, "DensePenaltyFunctionBase");
  python::BaseWrapper<Eigen::SparseMatrix<double> >(mod, "SparsePenaltyFunctionBase");

  py::class_<DensePenaltyFunction, PyDensePenaltyFunction, std::shared_ptr<DensePenaltyFunction>, PenaltyFunction<Eigen::MatrixXd> > dense(mod, "DensePenaltyFunction");
  dense.def(py::init<std::size_t const, std::size_t const, std::shared_ptr<const Parameters> const&>());
  dense.def(py::init<std::size_t const, std::size_t const>());

  dense.def("Evaluate", &DensePenaltyFunction::Evaluate);

  py::class_<SparsePenaltyFunction, PySparsePenaltyFunction, std::shared_ptr<SparsePenaltyFunction>, PenaltyFunction<Eigen::SparseMatrix<double> > > sparse(mod, "SparsePenaltyFunction");
  sparse.def(py::init<std::size_t const, std::size_t const, std::shared_ptr<const Parameters> const&>());
  sparse.def(py::init<std::size_t const, std::size_t const>());

  sparse.def("Evaluate", &SparsePenaltyFunction::Evaluate);

}
