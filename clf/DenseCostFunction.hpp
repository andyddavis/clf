#ifndef DENSECOSTFUNCTION_HPP_
#define DENSECOSTFUNCTION_HPP_

#include "clf/CostFunction.hpp"
#include "clf/DensePenaltyFunction.hpp"

namespace clf {

/// A cost function (see clf::CostFunction) that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt) using dense matrices
class DenseCostFunction : public CostFunction<Eigen::MatrixXd> {
public:
  
  /**
     @param[in] penaltyFuncs A vector of penalty functions such that the \f$i^{\text{th}}\f$ entry is \f$c_i: \mathbb{R}^{d} \mapsto \mathbb{R}^{n_i}\f$
  */
  DenseCostFunction(DensePenaltyFunctions const& penaltyFunctions);

  /**
     @param[in] penaltyFunction The penalty function \f$c_0: \mathbb{R}^{d} \mapsto \mathbb{R}^{n_i}\f$
  */
  DenseCostFunction(std::shared_ptr<DensePenaltyFunction> const& penaltyFunction);

  virtual ~DenseCostFunction() = default; 

  /// Compute the Jacobian of the penalty functions \f$\nabla_{\beta} c \in \mathbb{R}^{\bar{n} \times d}\f$
  /**
     The Jacobian of the penalty function is 
     \f{equation*}{
     \nabla_{\beta} c(\beta) = \begin{bmatrix}
     \nabla_{\beta} c_0(\beta) \\
     \vdots \\
     \nabla_{\beta} c_m(\beta) 
     \end{bmatrix}.
     \f}
     @param[in] beta The input parameters \f$\beta\f$
     \return The Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{\bar{n} \times d}\f$
  */
  virtual Eigen::MatrixXd Jacobian(Eigen::VectorXd const& beta) const final override;

private:
};

} // namespace clf 

#endif
