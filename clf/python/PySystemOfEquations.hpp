#include "clf/SystemOfEquations.hpp"

#include <pybind11/eigen.h>

namespace clf {
namespace python {

/// A trampoline class for the python interface of clf::SystemOfEquations
template<typename PARENT>
class PySystemOfEquations : public PARENT {

  // inherit the constructors
  using PARENT::PARENT;
  
  /// Evaluate the right hand side at the location \f$x\f$.
  /**
     Defaults to \f$f(x)=0\f$.
     @param[in] x The location \f$x\f$
     \return The right hand side \f$f(x)\f$
   */
  inline virtual Eigen::VectorXd RightHandSide(Eigen::VectorXd const& x) const { PYBIND11_OVERRIDE(Eigen::VectorXd, PARENT, RightHandSide, x); }
};

  
} // namespace python
} // namespace clf
  
