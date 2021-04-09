#ifndef SUPPORTPOINT_HPP_
#define SUPPORTPOINT_HPP_

#include <MUQ/Modeling/ModPiece.h>

namespace clf {

/// The local function associated with a support point \f$x\f$.
class SupportPoint {
public:

  /**
  @param[in] x The location of the support point \f$x\f$.
  */
  SupportPoint(Eigen::VectorXd const& x);

  virtual ~SupportPoint() = default;

  /// The location of the support point \f$x\f$.
  const Eigen::VectorXd x;

private:
};

} // namespace clf

#endif
