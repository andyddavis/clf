#ifndef LAPLACEOPERATOR_HPP_
#define LAPLACEOPERATOR_HPP_

#include "clf/LinearModel.hpp"

namespace clf {

/// A model for the Laplace operator \f$\mathcal{L}(u) = \nabla \cdot \nabla u\f$, where \f$u: \mathbb{R}^{d} \mapsto \mathbb{R}^{m}\f$.
/**
The local function is defined by \f$u(x) = \Phi_{\hat{x}}(x) p\f$ (see clf::SupportPoint), where \f$\Phi_{\hat{x}}(x) \in \mathbb{R}^{m \times \tilde{q}}\f$ and \f$p \in \mathbb{R}^{\tilde{q}}\f$. The Laplace operator is, therefore, defined by the matrix \f$L_{\hat{x}} \in \mathbb{R}^{m \times \tilde{q}}\f$ such that
\f{equation*}{
     \nabla \cdot \nabla u = \nabla \cdot \nabla ( \Phi_{\hat{x}}(x) p ) = \left[ \begin{array}{ccc|ccc|c|ccc}
        \nabla \cdot \nabla (\phi_{\hat{x}}^{(0)}(x))_0 & --- & \nabla \cdot \nabla (\phi_{\hat{x}}^{(0)}(x))_{q_0} & --- & 0 & --- & ... & --- & 0 & --- \\
        --- & 0 & --- & \nabla \cdot \nabla (\phi_{\hat{x}}^{(1)}(x))_0 & --- & \nabla \cdot \nabla (\phi_{\hat{x}}^{(1)}(x))_{q_1} & ... & --- & 0 & --- \\
        --- & \vdots & --- & --- & \vdots & --- & \ddots & --- & \vdots & --- \\
        --- & 0 & --- & --- & 0 & --- & ... &   \nabla \cdot \nabla (\phi_{\hat{x}}^{(m)}(x))_0 & --- & \nabla \cdot \nabla (\phi_{\hat{x}}^{(m)}(x))_{q_m}
     \end{array} \right] p = L_{\hat{x}} p
\f}
*/
class LaplaceOperator : public LinearModel {
public:

  /**
  @param[in] indim The input dimension \f$d\f$
  @param[in] outdim The output dimension \f$m\f$ (defaults to \f$1\f$)
  */
  LaplaceOperator(std::size_t const indim, std::size_t const outdim = 1);

  virtual ~LaplaceOperator() = default;

  /// The matrix \f$L_{\hat{x}}\f$ that defines the linear model
  /**
  \f{equation*}{
  L_{\hat{x}} = \left[ \begin{array}{ccc|ccc|c|ccc}
        \nabla \cdot \nabla (\phi_{\hat{x}}^{(0)}(x))_0 & --- & \nabla \cdot \nabla (\phi_{\hat{x}}^{(0)}(x))_{q_0} & --- & 0 & --- & ... & --- & 0 & --- \\
        --- & 0 & --- & \nabla \cdot \nabla (\phi_{\hat{x}}^{(1)}(x))_0 & --- & \nabla \cdot \nabla (\phi_{\hat{x}}^{(1)}(x))_{q_1} & ... & --- & 0 & --- \\
        --- & \vdots & --- & --- & \vdots & --- & \ddots & --- & \vdots & --- \\
        --- & 0 & --- & --- & 0 & --- & ... &   \nabla \cdot \nabla (\phi_{\hat{x}}^{(m)}(x))_0 & --- & \nabla \cdot \nabla (\phi_{\hat{x}}^{(m)}(x))_{q_m}
     \end{array} \right] \in \mathbb{R}^{m \times \tilde{q}}
  \f}
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] bases The basis functions for each output
  \return The matrix \f$L_{\hat{x}}(x)\f$ that defines the linear model
   */
  virtual Eigen::MatrixXd ModelMatrix(Eigen::VectorXd const& x, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override;

private:
};

} // namespace clf

#endif
