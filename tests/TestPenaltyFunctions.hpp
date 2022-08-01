#ifndef TESTPENALTYFUNCTIONS_HPP_
#define TESTPENALTYFUNCTIONS_HPP_

#include "clf/PenaltyFunction.hpp"

namespace clf {
namespace tests {

/// An example penalty function used to test clf::DensePenaltyFunction
class DensePenaltyFunctionTest0 : public DensePenaltyFunction {
public:
  
  inline DensePenaltyFunctionTest0() : DensePenaltyFunction(3, 2) {}

  virtual ~DensePenaltyFunctionTest0() = default;

  /// Evaluate the penalty function \f$c: \mathbb{R}^{d} \mapsto \mathbb{R}^{n}\f$
  /**
     The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the penalty function is
     \f{equation*}{
     c(\beta) = \begin{bmatrix}
     \beta_0 \\
     \beta_0 (1-\beta_2)
     \end{bmatrix}.
     \f}
     @param[in] beta The input parameters \f$\beta\f$
     \return A vector \f$c \in \mathbb{R}^{m}\f$ such that the \f$i^{\text{th}}\f$ entry is \f$c_i(\beta)\f$
     @param[in] beta The input parameters \f$\beta\f$
     \return The penalty function evaluation \f$c(\beta)\f$
   */
  inline virtual Eigen::VectorXd Evaluate(Eigen::VectorXd const& beta) override { 
    assert(beta.size()==3);
    return Eigen::Vector2d(beta(0), beta(0)*(1.0-beta(2))); 
  }

  /// Evaluate the Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{2 \times 3}\f$
  /**
     The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the penalty function Jacobian is
     \f{equation*}{
     \nabla_{\beta} c(\beta) = \begin{bmatrix}
     1 & 0 & 0 \\
     1-\beta_2 & 0 & -\beta_0 
     \end{bmatrix} \in \mathbb{R}^{2 \times 3}.
     \f}
     @param[in] beta The input parameters \f$\beta\f$
     \return The gradient of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{2 \times 3}\f$
  */
  inline virtual Eigen::MatrixXd Jacobian(Eigen::VectorXd const& beta) override { 
    Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(2, 3);
    jac(0, 0) = 1.0;
    jac(1, 0) = 1.0-beta(2); jac(1, 2) = -beta(0);

    return jac;
  }

  /// Compute the sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
  /**
     The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the Hessian of the first component of the the penalty function is
     \f{equation*}{
     \nabla_{\beta}^2 c_1(\beta) = \begin{bmatrix}
     0 & 0 & 0 \\
     0 & 0 & 0 \\
     0 & 0 & 0
     \end{bmatrix} \in \mathbb{R}^{3 \times 3}
     \f}
     and the Hessian of the second component of the the penalty function is
     \f{equation*}{
     \nabla_{\beta}^2 c_2(\beta) = \begin{bmatrix}
     0 & 0 & -1 \\ 
     0 & 0 & 0 \\ 
     -1 & 0 & 0 
     \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
     \f}
     This returns the sum of these Hessians.
     @param[in] beta The input parameters \f$\beta\f$
     @param[in] weights The weights for the weighted sum
     \return The sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
  */
  inline virtual Eigen::MatrixXd Hessian(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) override { 
    assert(beta.size()==3);
    
    Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(3, 3);
    hess(0, 2) += -weights(1); hess(2, 0) += -weights(1);

    return hess;
  }
  
private:
};

/// An example penalty function used to test clf::DensePenaltyFunction
class DensePenaltyFunctionTest1 : public DensePenaltyFunction {
public:
  
  inline DensePenaltyFunctionTest1() : DensePenaltyFunction(3, 6) {}

  virtual ~DensePenaltyFunctionTest1() = default;

  /// Evaluate the penalty function \f$c: \mathbb{R}^{d} \mapsto \mathbb{R}^{n}\f$
  /**
     The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the penalty function is
     \f{equation*}{
     c(\beta) = \begin{bmatrix}
     1-\beta_1 \\
     1-\beta_1+\beta_2 \\
     \beta_2 \\
     \beta_2(1-\beta_1) \\
     \beta_0 \beta_2 \\
     \beta_0^2 \beta_1
     \end{bmatrix}.
     \f}
     @param[in] beta The input parameters \f$\beta\f$
     \return A vector \f$c \in \mathbb{R}^{m}\f$ such that the \f$i^{\text{th}}\f$ entry is \f$c_i(\beta)\f$
     @param[in] beta The input parameters \f$\beta\f$
     \return The penalty function evaluation \f$c(\beta)\f$
   */
  inline virtual Eigen::VectorXd Evaluate(Eigen::VectorXd const& beta) override { 
    assert(beta.size()==3);

    Eigen::VectorXd output(outdim);
    output(0) = 1.0-beta(1);
    output(1) = 1.0-beta(1)+beta(2);
    output(2) = beta(2);
    output(3) = beta(2)*(1.0-beta(1));
    output(4) = beta(0)*beta(2);
    output(5) = beta(0)*beta(0)*beta(1);

    return output;
  }

  /// Evaluate the Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{6 \times 3}\f$
  /**
     The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the penalty function Jacobian is
     \f{equation*}{
     \nabla_{\beta} c(\beta) = \begin{bmatrix}
     0 & -1 & 0 \\
     0 & -1 & 1 \\
     0 & 0 & 1 \\
     0 & -\beta_2 & 1-\beta_1 \\
     \beta_2 & 0 & \beta_0 \\
     2 \beta_0 \beta_1 & \beta_0^2 & 0
     \end{bmatrix} \in \mathbb{R}^{6 \times 3}.
     \f}
     @param[in] beta The input parameters \f$\beta\f$
     \return The Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{2 \times 3}\f$
  */
  inline virtual Eigen::MatrixXd Jacobian(Eigen::VectorXd const& beta) override { 
    assert(beta.size()==3);
    Eigen::MatrixXd jac(6, 3);
    jac.row(0) << 0.0, -1.0, 0.0;
    jac.row(1) << 0.0, -1.0, 1.0;
    jac.row(2) << 0.0, 0.0, 1.0;
    jac.row(3) << 0.0, -beta(2), 1.0-beta(1);
    jac.row(4) << beta(2), 0.0, beta(0);
    jac.row(5) << 2.0*beta(0)*beta(1), beta(0)*beta(0), 0.0;

    return jac;
  }

  /// Compute the sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
  /**
     The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the Hessian of the \f$1^{\text{st}}\f$ component of the the penalty function is
     \f{equation*}{
     \nabla_{\beta}^2 c_1(\beta) = \begin{bmatrix}
     0 & 0 & 0 \\
     0 & 0 & 0 \\
     0 & 0 & 0
     \end{bmatrix} \in \mathbb{R}^{3 \times 3}
     \f}
     and the Hessian of the \f$2^{\text{nd}}\f$ component of the the penalty function is
     \f{equation*}{
     \nabla_{\beta}^2 c_2(\beta) = \begin{bmatrix}
     0 & 0 & 0 \\ 
     0 & 0 & 0 \\ 
     0 & 0 & 0 
     \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
     \f}
     and the Hessian of the \f$3^{\text{rd}}\f$ component of the the penalty function is
     \f{equation*}{
     \nabla_{\beta}^2 c_3(\beta) = \begin{bmatrix}
     0 & 0 & 0 \\ 
     0 & 0 & 0 \\ 
     0 & 0 & 0 
     \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
     \f}
     and the Hessian of the \f$4^{\text{th}}\f$ component of the the penalty function is
     \f{equation*}{
     \nabla_{\beta}^2 c_4(\beta) = \begin{bmatrix}
     0 & 0 & 0 \\ 
     0 & 0 & -1 \\ 
     0 & -1 & 0 
     \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
     \f}
     and the Hessian of the \f$5^{\text{th}}\f$ component of the the penalty function is
     \f{equation*}{
     \nabla_{\beta}^2 c_5(\beta) = \begin{bmatrix}
     0 & 0 & 1 \\ 
     0 & 0 & 0 \\ 
     1 & 0 & 0 
     \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
     \f}
     and the Hessian of the \f$6^{\text{th}}\f$ component of the the penalty function is
     \f{equation*}{
     \nabla_{\beta}^2 c_6(\beta) = \begin{bmatrix}
     2 \beta_1 & 2 \beta_0 & 0 \\ 
     2 \beta_0 & 0 & 0 \\ 
     0 & 0 & 0 
     \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
     \f}
     This returns the sum of these Hessians.
     @param[in] beta The input parameters \f$\beta\f$
     \return The sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
  */
  inline virtual Eigen::MatrixXd Hessian(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) override { 
    assert(beta.size()==3);
    
    Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(3, 3);
    hess(1, 2) += -weights(3); hess(2, 1) += -weights(3);
    hess(0, 2) += weights(4); hess(2, 0) += weights(4);
    hess(0, 0) += 2.0*beta(1)*weights(5); hess(0, 1) += 2.0*beta(0)*weights(5); hess(1, 0) += 2.0*beta(0)*weights(5);

    return hess;
  }

private:
};

/// An example penalty function used to test clf::SparsePenaltyFunction
class SparsePenaltyFunctionTest0 : public SparsePenaltyFunction {
public:
  
  inline SparsePenaltyFunctionTest0() : SparsePenaltyFunction(3, 2) {}

  virtual ~SparsePenaltyFunctionTest0() = default;

  /// Evaluate the penalty function \f$c: \mathbb{R}^{d} \mapsto \mathbb{R}^{n}\f$
  /**
     The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the penalty function is
     \f{equation*}{
     c(\beta) = \begin{bmatrix}
     \beta_0 \\
     \beta_0 (1-\beta_2)
     \end{bmatrix}.
     \f}
     @param[in] beta The input parameters \f$\beta\f$
     \return A vector \f$c \in \mathbb{R}^{m}\f$ such that the \f$i^{\text{th}}\f$ entry is \f$c_i(\beta)\f$
     @param[in] beta The input parameters \f$\beta\f$
     \return The penalty function evaluation \f$c(\beta)\f$
   */
  inline virtual Eigen::VectorXd Evaluate(Eigen::VectorXd const& beta) override { 
    assert(beta.size()==3);
    return Eigen::Vector2d(beta(0), beta(0)*(1.0-beta(2))); 
  }

  /// Evaluate the Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{2 \times 3}\f$
  /**
     The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the penalty function Jacobian is
     \f{equation*}{
     \nabla_{\beta} c(\beta) = \begin{bmatrix}
     1 & 0 & 0 \\
     1-\beta_2 & 0 & -\beta_0 
     \end{bmatrix} \in \mathbb{R}^{2 \times 3}.
     \f}
     @param[in] beta The input parameters \f$\beta\f$
     @param[out] entries The entries of the Jacobian matrix
  */
  inline virtual void JacobianEntries(Eigen::VectorXd const& beta, std::vector<Eigen::Triplet<double> >& entries) override { 
    assert(beta.size()==3);

    entries.resize(3);
    entries[0] = Eigen::Triplet<double>(0, 0, 1.0);
    entries[1] = Eigen::Triplet<double>(1, 0, 1.0-beta(2));
    entries[2] = Eigen::Triplet<double>(1, 2, -beta(0));
  }

  /// Compute the sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
  /**
     The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the Hessian of the first component of the the penalty function is
     \f{equation*}{
     \nabla_{\beta}^2 c_1(\beta) = \begin{bmatrix}
     0 & 0 & 0 \\
     0 & 0 & 0 \\
     0 & 0 & 0
     \end{bmatrix} \in \mathbb{R}^{3 \times 3}
     \f}
     and the Hessian of the second component of the the penalty function is
     \f{equation*}{
     \nabla_{\beta}^2 c_2(\beta) = \begin{bmatrix}
     0 & 0 & -1 \\ 
     0 & 0 & 0 \\ 
     -1 & 0 & 0 
     \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
     \f}
     This returns the sum of these Hessians.
     @param[in] beta The input parameters \f$\beta\f$
     @param[in] weights The weights for the weighted sum
     @param[out] entries The entries of the Hessian matrix
  */
  inline virtual void HessianEntries(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights, std::vector<Eigen::Triplet<double> >& entries) override { 
    assert(beta.size()==3);
    assert(weights.size()==2);

    entries.resize(2);
    entries[0] = Eigen::Triplet<double>(0, 2, -weights(1));
    entries[1] = Eigen::Triplet<double>(2, 0, -weights(1));
  }

private:
};

/// An example penalty function used to test clf::SparsePenaltyFunction
class SparsePenaltyFunctionTest1 : public SparsePenaltyFunction {
public:
  
  inline SparsePenaltyFunctionTest1() : SparsePenaltyFunction(3, 6) {}

  virtual ~SparsePenaltyFunctionTest1() = default;

  /// Evaluate the penalty function \f$c: \mathbb{R}^{d} \mapsto \mathbb{R}^{n}\f$
  /**
     The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the penalty function is
     \f{equation*}{
     c(\beta) = \begin{bmatrix}
     1-\beta_1 \\
     1-\beta_1+\beta_2 \\
     \beta_2 \\
     \beta_2(1-\beta_1) \\
     \beta_0 \beta_2 \\
     \beta_0^2 \beta_1
     \end{bmatrix}.
     \f}
     @param[in] beta The input parameters \f$\beta\f$
     \return A vector \f$c \in \mathbb{R}^{m}\f$ such that the \f$i^{\text{th}}\f$ entry is \f$c_i(\beta)\f$
     @param[in] beta The input parameters \f$\beta\f$
     \return The penalty function evaluation \f$c(\beta)\f$
   */
  inline virtual Eigen::VectorXd Evaluate(Eigen::VectorXd const& beta) override { 
    assert(beta.size()==3);

    Eigen::VectorXd output(outdim);
    output(0) = 1.0-beta(1);
    output(1) = 1.0-beta(1)+beta(2);
    output(2) = beta(2);
    output(3) = beta(2)*(1-beta(1));
    output(4) = beta(0)*beta(2);
    output(5) = beta(0)*beta(0)*beta(1);

    return output;
  }

  /// Evaluate the Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{6 \times 3}\f$
  /**
     The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the penalty function Jacobian is
     \f{equation*}{
     \nabla_{\beta} c(\beta) = \begin{bmatrix}
     0 & -1 & 0 \\
     0 & -1 & 1 \\
     0 & 0 & 1 \\
     0 & -\beta_2 & 1-\beta_1 \\
     \beta_2 & 0 & \beta_0 \\
     2 \beta_0 \beta_1 & \beta_0^2 & 0
     \end{bmatrix} \in \mathbb{R}^{6 \times 3}.
     \f}
     @param[in] beta The input parameters \f$\beta\f$
     @param[out] entries The entries of the Jacobian matrix
  */
  inline virtual void JacobianEntries(Eigen::VectorXd const& beta, std::vector<Eigen::Triplet<double> >& entries) override { 
    assert(beta.size()==3);

    entries.resize(10);
    entries[0] = Eigen::Triplet<double>(0, 1, -1.0);
    entries[1] = Eigen::Triplet<double>(1, 1, -1.0);
    entries[2] = Eigen::Triplet<double>(1, 2, 1.0);
    entries[3] = Eigen::Triplet<double>(2, 2, 1.0);
    entries[4] = Eigen::Triplet<double>(3, 1, -beta(2));
    entries[5] = Eigen::Triplet<double>(3, 2, 1.0-beta(1));
    entries[6] = Eigen::Triplet<double>(4, 0, beta(2));
    entries[7] = Eigen::Triplet<double>(4, 2, beta(0));
    entries[8] = Eigen::Triplet<double>(5, 0, 2.0*beta(0)*beta(1));
    entries[9] = Eigen::Triplet<double>(5, 1, beta(0)*beta(0));
  }

  /// Compute the sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
  /**
     The input dimension is \f$3\f$ (i.e., \f$d=3\f$ and \f$\beta \in \mathbb{R}^{3}\f$) and the Hessian of the \f$1^{\text{st}}\f$ component of the the penalty function is
     \f{equation*}{
     \nabla_{\beta}^2 c_1(\beta) = \begin{bmatrix}
     0 & 0 & 0 \\
     0 & 0 & 0 \\
     0 & 0 & 0
     \end{bmatrix} \in \mathbb{R}^{3 \times 3}
     \f}
     and the Hessian of the \f$2^{\text{nd}}\f$ component of the the penalty function is
     \f{equation*}{
     \nabla_{\beta}^2 c_2(\beta) = \begin{bmatrix}
     0 & 0 & 0 \\ 
     0 & 0 & 0 \\ 
     0 & 0 & 0 
     \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
     \f}
     and the Hessian of the \f$3^{\text{rd}}\f$ component of the the penalty function is
     \f{equation*}{
     \nabla_{\beta}^2 c_3(\beta) = \begin{bmatrix}
     0 & 0 & 0 \\ 
     0 & 0 & 0 \\ 
     0 & 0 & 0 
     \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
     \f}
     and the Hessian of the \f$4^{\text{th}}\f$ component of the the penalty function is
     \f{equation*}{
     \nabla_{\beta}^2 c_4(\beta) = \begin{bmatrix}
     0 & 0 & 0 \\ 
     0 & 0 & -1 \\ 
     0 & -1 & 0 
     \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
     \f}
     and the Hessian of the \f$5^{\text{th}}\f$ component of the the penalty function is
     \f{equation*}{
     \nabla_{\beta}^2 c_5(\beta) = \begin{bmatrix}
     0 & 0 & 1 \\ 
     0 & 0 & 0 \\ 
     1 & 0 & 0 
     \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
     \f}
     and the Hessian of the \f$6^{\text{th}}\f$ component of the the penalty function is
     \f{equation*}{
     \nabla_{\beta}^2 c_6(\beta) = \begin{bmatrix}
     2 \beta_1 & 2 \beta_0 & 0 \\ 
     2 \beta_0 & 0 & 0 \\ 
     0 & 0 & 0 
     \end{bmatrix} \in \mathbb{R}^{3 \times 3}.
     \f}
     This returns the sum of these Hessians.
     @param[in] beta The input parameters \f$\beta\f$
     @param[in] weights The weights for the weighted sum
     @param[out] entries The entries of the Hessian matrix
  */
  inline virtual void HessianEntries(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights, std::vector<Eigen::Triplet<double> >& entries) override { 
    assert(beta.size()==3);

    entries.resize(7);
    entries[0] = Eigen::Triplet<double>(1, 2, -weights(3));
    entries[1] = Eigen::Triplet<double>(2, 1, -weights(3));
    entries[2] = Eigen::Triplet<double>(0, 2, weights(4));
    entries[3] = Eigen::Triplet<double>(2, 0, weights(4));
    entries[4] = Eigen::Triplet<double>(0, 0, 2.0*weights(5)*beta(1));
    entries[5] = Eigen::Triplet<double>(0, 1, 2.0*weights(5)*beta(0));
    entries[6] = Eigen::Triplet<double>(1, 0, 2.0*weights(5)*beta(0));
  }

private:
};

} // namespace tests
} // namespace clf

#endif


