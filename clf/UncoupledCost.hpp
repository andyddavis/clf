#ifndef UNCOUPLEDCOST_HPP_
#define UNCOUPLEDCOST_HPP_

#include <MUQ/Optimization/CostFunction.h>

namespace clf {

/// Forward declaration of a support point
class SupportPoint;

/// Compute the uncoupled cost associated with a support point
/**
The uncoupled cost function associated with support point \f$i\f$ is
\f{equation*}{
p_i = \mbox{arg min}_{p \in \mathbb{R}^{\bar{q}_i}} J(p) = \sum_{j=1}^{k_{nn}} \frac{m_i}{2} \| \mathcal{L}_i(\hat{u}(x_{I(i,j)}, p)) - f_i(x_{I(i,j)}) \|^2 {K_i(x_i, x_{I(i,j)})} + \frac{a_i}{2} \|p\|^2,
\f}
*/
class UncoupledCost : public muq::Optimization::CostFunction {
public:

  UncoupledCost(std::shared_ptr<SupportPoint> const& point);

  virtual ~UncoupledCost() = default;

  Eigen::MatrixXd Hessian(Eigen::VectorXd const& coefficients, bool const gaussNewtonHessian) const;

  /// The point that is associated with this cost
  std::weak_ptr<const SupportPoint> point;

protected:

  virtual double CostImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& input) override;

  virtual void GradientImpl(unsigned int const inputDimWrt, muq::Modeling::ref_vector<Eigen::VectorXd> const& input, Eigen::VectorXd const& sensitivity) override;

private:
};

} // namespace clf

#endif
