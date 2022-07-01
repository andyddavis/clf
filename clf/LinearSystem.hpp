#ifndef LINEARSYSTEM_HPP_
#define LINEARSYSTEM_HPP_

#include "clf/SystemOfEquations.hpp"

namespace clf {

/// A system of equations with the form \f$A(x) u(x) - f(x) = 0\f$, where \f$u(x) = \Phi(x)^{\top} c\f$.
class LinearSystem : public SystemOfEquations {
public:

  LinearSystem();

  virtual ~LinearSystem() = default;

private:
};

} // namespace clf

#endif
