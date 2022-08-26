#include "clf/python/Pybind11Wrappers.hpp"

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "clf/SparsePenaltyFunction.hpp"

namespace py = pybind11;

namespace clf {
namespace python {

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

    /// Compute the terms of the sparse Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$ using
  /**
     @param[in] beta The input parameters \f$\beta\f$
     \return The entries of the Jacobian matrix
   */
  virtual std::vector<Eigen::Triplet<double> > JacobianEntries(Eigen::VectorXd const& beta) override { PYBIND11_OVERRIDE(std::vector<Eigen::Triplet<double> >, SparsePenaltyFunction, JacobianEntries, beta); }
  
    /// Compute entries of the weighted sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} w_i \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$ 
  /**
     @param[in] beta The input parameters \f$\beta\f$
     @param[in] weights The weights for the weighted sum
     \return The entries of the Hessian matrix
   */
  virtual std::vector<Eigen::Triplet<double> > HessianEntries(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) override { PYBIND11_OVERRIDE(std::vector<Eigen::Triplet<double> >, SparsePenaltyFunction, HessianEntries, beta, weights); }

private:
};

} // namespace python
} // namespace clf

using namespace clf;

void clf::python::SparsePenaltyFunctionWrapper(py::module& mod) {    
  py::class_<SparsePenaltyFunction, PySparsePenaltyFunction, std::shared_ptr<SparsePenaltyFunction>, PenaltyFunction<Eigen::SparseMatrix<double> > > sparse(mod, "SparsePenaltyFunction");
  sparse.def(py::init<std::size_t const, std::size_t const, std::shared_ptr<const Parameters> const&>());
  sparse.def(py::init<std::size_t const, std::size_t const>());

  sparse.def("Evaluate", &SparsePenaltyFunction::Evaluate);
  sparse.def("Jacobian", &SparsePenaltyFunction::Jacobian);
  sparse.def("JacobianEntries", &SparsePenaltyFunction::JacobianEntries);
  sparse.def("Hessian", &SparsePenaltyFunction::Hessian);
  sparse.def("HessianEntries", &SparsePenaltyFunction::HessianEntries);
}
