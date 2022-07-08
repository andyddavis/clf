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
    output(3) = beta(2)*(1-beta(1));
    output(4) = beta(0)*beta(2);
    output(5) = beta(0)*beta(0)*beta(1);

    return output;
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

private:
};

} // namespace tests
} // namespace clf

#endif


