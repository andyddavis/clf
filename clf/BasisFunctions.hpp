#ifndef BASISFUNCTIONS_HPP_
#define BASISFUNCTIONS_HPP_

namespace clf {

/// Evaluate the basis functions \f$\phi(x) = [\phi_1(x),\, \phi_2(x),\, ...,\, \phi_q(x)]\f$
class BasisFunctions {
public:
  BasisFunctions();

  virtual ~BasisFunctions() = default;
private:
};

} // namespace clf

#endif
