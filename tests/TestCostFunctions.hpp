#ifndef TESTCOSTFUNCTIONS_HPP_
#define TESTCOSTFUNCTIONS_HPP_

#include "clf/DenseCostFunction.hpp"
#include "clf/SparseCostFunction.hpp"
#include "clf/DenseQuadraticCostFunction.hpp"
#include "clf/SparseQuadraticCostFunction.hpp"

namespace clf {
namespace tests {

/// An example cost function used to test clf::DenseCostFunction
/**
The input dimension is \f$3\f$ and the number of penalty functions is \f$4\f$. The penalty functions are
\f{equation*}{
\begin{array}{ccccc}
f_0(\beta) = \left[ \begin{array}{c}
\beta_0 \\
\beta_0 (1-\beta_2)
\end{array} \right], &
f_1(\beta) = \left[ \begin{array}{c}
1-\beta_1 \\
1-\beta_1+\beta_2
\end{array} \right], &
f_2(\beta) = \left[ \begin{array}{c}
\beta_2 \\
\beta_2 (1-\beta_1)
\end{array} \right], &
\mbox{and} &
f_3(\beta) = \left[ \begin{array}{c}
\beta_0 \beta_2 \\
\beta_0^2 \beta_1
\end{array} \right].
\end{array}
\f}
*/
class DenseCostFunctionTest : public DenseCostFunction {
public:

  /// The input dimension is \f$3\f$ and the number of penalty terms is \f$4\f$, each with dimension \f$2\f$
  inline DenseCostFunctionTest() : DenseCostFunction(3, 4, 2) {}

  virtual ~DenseCostFunctionTest() = default;

protected:

  /// Implement the pentalty functions
  /**
  The penalty functions are
  \f{equation*}{
  \begin{array}{ccccc}
  f_0(\beta) = \left[ \begin{array}{c}
  \beta_0 \\
  \beta_0 (1-\beta_2)
  \end{array} \right], &
  f_1(\beta) = \left[ \begin{array}{c}
  1-\beta_1 \\
  1-\beta_1+\beta_2
  \end{array} \right], &
  f_2(\beta) = \left[ \begin{array}{c}
  \beta_2 \\
  \beta_2 (1-\beta_1)
  \end{array} \right], &
  \mbox{and} &
  f_3(\beta) = \left[ \begin{array}{c}
  \beta_0 \beta_2 \\
  \beta_0^2 \beta_1
  \end{array} \right].
  \end{array}
  \f}
  @param[in] ind The index of the penalty function we are implementing
  @param[in] beta The parameter value
  \return The \f$i^{th}\f$ penalty function \f$f_i(\beta)\f$
  */
  inline virtual Eigen::VectorXd PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const override {
    switch( ind ) {
      case 0:
      return Eigen::Vector2d(beta(0), beta(0)*(1.0-beta(2)));
      case 1:
      return Eigen::Vector2d(1.0-beta(1), 1.0-beta(1)+beta(2));
      case 2:
      return Eigen::Vector2d(beta(2), beta(2)*(1.0-beta(1)));
      case 3:
      return Eigen::Vector2d(beta(0)*beta(2), beta(0)*beta(0)*beta(1));
    }
    return Eigen::VectorXd::Zero(2);
  }

  /// Implement the pentalty function gradients
  /**
  The penalty functions are
  \f{equation*}{
  \begin{array}{ccccc}
  f_0(\beta) = \left[ \begin{array}{c}
  \beta_0 \\
  \beta_0 (1-\beta_2)
  \end{array} \right], &
  f_1(\beta) = \left[ \begin{array}{c}
  1-\beta_1 \\
  1-\beta_1+\beta_2
  \end{array} \right], &
  f_2(\beta) = \left[ \begin{array}{c}
  \beta_2 \\
  \beta_2 (1-\beta_1)
  \end{array} \right], &
  \mbox{and} &
  f_3(\beta) = \left[ \begin{array}{c}
  \beta_0 \beta_2 \\
  \beta_0^2 \beta_1
  \end{array} \right].
  \end{array}
  \f}
  @param[in] ind The index of the penalty function we are implementing
  @param[in] beta The parameter value
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$
  */
  inline virtual Eigen::MatrixXd PenaltyFunctionJacobianImpl(std::size_t const ind, Eigen::VectorXd const& beta) const override {
    Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(2, 3);
    switch( ind ) {
      case 0:
      jac(0, 0) = 1;
      jac(1, 0) = 1.0-beta(2); jac(1, 2) = -beta(0);
      break;
      case 1:
      jac(0, 1) = -1.0;
      jac(1, 1) = -1.0; jac(1, 2) = 1.0;
      break;
      case 2:
      jac(0, 2) = 1.0;
      jac(1, 1) = -beta(2); jac(1, 2) = 1.0-beta(1);
      break;
      case 3:
      jac(0, 0) = beta(2); jac(0, 2) = beta(0);
      jac(1, 0) = 2.0*beta(0)*beta(1); jac(1, 1) = beta(0)*beta(0);
      break;
    }
    return jac;
  }

private:
};

/// An example cost function used to test clf::DenseCostFunction
/**
The input dimension is \f$3\f$ and the number of penalty functions is \f$4\f$. The penalty functions are
\f{equation*}{
\begin{array}{ccccc}
f_0(\beta) = \left[ \begin{array}{c}
\beta_0 \\
\beta_0 (1-\beta_2)
\end{array} \right], &
f_1(\beta) = \left[ \begin{array}{c}
1-\beta_1 \\
1-\beta_1+\beta_2
\end{array} \right], &
f_2(\beta) = \left[ \begin{array}{c}
\beta_2 \\
\beta_2 (1-\beta_1)
\end{array} \right], &
\mbox{and} &
f_3(\beta) = \left[ \begin{array}{c}
\beta_0 \beta_2 \\
\beta_0^2 \beta_1
\end{array} \right].
\end{array}
\f}
*/
class SparseCostFunctionTest : public SparseCostFunction {
public:

  /// The input dimension is \f$3\f$ and the number of penalty terms is \f$4\f$, each with dimension \f$2\f$
  inline SparseCostFunctionTest() : SparseCostFunction(3, 4, 2) {}

  virtual ~SparseCostFunctionTest() = default;

protected:

  /// Implement the pentalty functions
  /**
  The penalty functions are
  \f{equation*}{
  \begin{array}{ccccc}
  f_0(\beta) = \left[ \begin{array}{c}
  \beta_0 \\
  \beta_0 (1-\beta_2)
  \end{array} \right], &
  f_1(\beta) = \left[ \begin{array}{c}
  1-\beta_1 \\
  1-\beta_1+\beta_2
  \end{array} \right], &
  f_2(\beta) = \left[ \begin{array}{c}
  \beta_2 \\
  \beta_2 (1-\beta_1)
  \end{array} \right], &
  \mbox{and} &
  f_3(\beta) = \left[ \begin{array}{c}
  \beta_0 \beta_2 \\
  \beta_0^2 \beta_1
  \end{array} \right].
  \end{array}
  \f}
  @param[in] ind The index of the penalty function we are implementing
  @param[in] beta The parameter value
  \return The \f$i^{th}\f$ penalty function \f$f_i(\beta)\f$
  */
  inline virtual Eigen::VectorXd PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const override {
    switch( ind ) {
      case 0:
      return Eigen::Vector2d(beta(0), beta(0)*(1.0-beta(2)));
      case 1:
      return Eigen::Vector2d(1.0-beta(1), 1.0-beta(1)+beta(2));
      case 2:
      return Eigen::Vector2d(beta(2), beta(2)*(1.0-beta(1)));
      case 3:
      return Eigen::Vector2d(beta(0)*beta(2), beta(0)*beta(0)*beta(1));
    }
    return Eigen::VectorXd::Zero(2);
  }

  /// Implement the pentalty function gradients
  /**
  The penalty functions are
  \f{equation*}{
  \begin{array}{ccccc}
  f_0(\beta) = \left[ \begin{array}{c}
  \beta_0 \\
  \beta_0 (1-\beta_2)
  \end{array} \right], &
  f_1(\beta) = \left[ \begin{array}{c}
  1-\beta_1 \\
  1-\beta_1+\beta_2
  \end{array} \right], &
  f_2(\beta) = \left[ \begin{array}{c}
  \beta_2 \\
  \beta_2 (1-\beta_1)
  \end{array} \right], &
  \mbox{and} &
  f_3(\beta) = \left[ \begin{array}{c}
  \beta_0 \beta_2 \\
  \beta_0^2 \beta_1
  \end{array} \right].
  \end{array}
  \f}
  @param[in] ind The index of the penalty function we are implementing
  @param[in] beta The parameter value
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$, each entry holds the index and value of a non-zero entry
  */
  inline virtual std::vector<std::pair<std::size_t, double> > PenaltyFunctionJacobianSparseImpl(std::size_t const ind, Eigen::VectorXd const& beta) const override {
    switch( ind ) {
      case 0:
      return std::vector<std::pair<std::size_t, double> >({std::pair<std::size_t, double>(0, 1.0)});
      case 1:
      return std::vector<std::pair<std::size_t, double> >({std::pair<std::size_t, double>(1, -1.0)});
      case 2:
      return std::vector<std::pair<std::size_t, double> >({std::pair<std::size_t, double>(2, 1.0)});
      case 3:
      return std::vector<std::pair<std::size_t, double> >({std::pair<std::size_t, double>(0, beta(2)), std::pair<std::size_t, double>(2, beta(0))});
    }
    return std::vector<std::pair<std::size_t, double> >();
  }

private:
};

/// An example cost function used to test clf::DenseCostFunction
/**
The input dimension is \f$3\f$ and the number of penalty functions is \f$4\f$. The penalty functions are
\f{equation*}{
\begin{array}{ccccc}
f_0(\beta) = \beta_0, & f_1(\beta) = 1-\beta_1, & f_2(\beta) = \beta_2 + \beta_1, & \mbox{and} & f_3(\beta) = 3 \beta_2.
\end{array}
\f}
*/
class DenseQuadraticCostTest : public DenseQuadraticCostFunction {
public:

  /// The input dimension is \f$3\f$ and the number of penalty terms is \f$4\f$
  inline DenseQuadraticCostTest() : DenseQuadraticCostFunction(3, 4, 1) {}

  virtual ~DenseQuadraticCostTest() = default;

protected:

  /// Implement the pentalty functions
  /**
  The penalty functions are
  \f{equation*}{
  \begin{array}{ccccc}
  f_0(\beta) = \beta_0, & f_1(\beta) = 1-\beta_1, & f_2(\beta) = \beta_2 + \beta_1, & \mbox{and} & f_3(\beta) = 3 \beta_2.
  \end{array}
  \f}
  @param[in] ind The index of the penalty function we are implementing
  @param[in] beta The paramAeter value
  \return The \f$i^{th}\f$ penalty function \f$f_i(\beta)\f$
  */
  inline virtual Eigen::VectorXd PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const override {
    switch( ind ) {
      case 0:
      return Eigen::VectorXd::Constant(1, beta(0));
      case 1:
      return Eigen::VectorXd::Constant(1, 1.0-beta(1));
      case 2:
      return Eigen::VectorXd::Constant(1, beta(2)+beta(1));
      case 3:
      return Eigen::VectorXd::Constant(1, 3.0*beta(2));
    }
    return Eigen::VectorXd::Zero(1);
  }

  /// Implement the pentalty function gradients
  /**
  The penalty functions are
  \f{equation*}{
  \begin{array}{ccccc}
  f_0(\beta) = \beta_0, & f_1(\beta) = 1-\beta_1, & f_2(\beta) = \beta_2 + \beta_1, & \mbox{and} & f_3(\beta) = 3 \beta_2.
  \end{array}
  \f}
  @param[in] ind The index of the penalty function we are implementing
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$
  */
  inline virtual Eigen::MatrixXd PenaltyFunctionJacobianImpl(std::size_t const ind) const override {
    switch( ind ) {
      case 0:
      return Eigen::Vector3d(1.0, 0.0, 0.0);
      case 1:
      return Eigen::Vector3d(0.0, -1.0, 0.0);
      case 2:
      return Eigen::Vector3d(0.0, 1.0, 1.0);
      case 3:
      return Eigen::Vector3d(0.0, 0.0, 3.0);
    }
    return Eigen::VectorXd::Zero(3);
  }

private:
};

/// An example cost function used to test clf::DenseCostFunction
/**
The input dimension is \f$3\f$ and the number of penalty functions is \f$4\f$. The penalty functions are
\f{equation*}{
\begin{array}{ccccc}
f_0(\beta) = \beta_0, & f_1(\beta) = 1-\beta_1, & f_2(\beta) = \beta_2 + \beta_1, & \mbox{and} & f_3(\beta) = 3 \beta_2.
\end{array}
\f}
*/
class SparseQuadraticCostTest : public SparseQuadraticCostFunction {
public:

  /// The input dimension is \f$3\f$ and the number of penalty terms is \f$4\f$
  inline SparseQuadraticCostTest() : SparseQuadraticCostFunction(3, 4, 1) {}

  virtual ~SparseQuadraticCostTest() = default;

protected:

  /// Implement the pentalty functions
  /**
  The penalty functions are
  \f{equation*}{
  \begin{array}{ccccc}
  f_0(\beta) = \beta_0, & f_1(\beta) = 1-\beta_1, & f_2(\beta) = \beta_2 + \beta_1, & \mbox{and} & f_3(\beta) = 3 \beta_2.
  \end{array}
  \f}
  @param[in] ind The index of the penalty function we are implementing
  @param[in] beta The parameter value
  \return The \f$i^{th}\f$ penalty function \f$f_i(\beta)\f$
  */
  inline virtual Eigen::VectorXd PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const override {
    switch( ind ) {
      case 0:
      return Eigen::VectorXd::Constant(1, beta(0));
      case 1:
      return Eigen::VectorXd::Constant(1, 1.0-beta(1));
      case 2:
      return Eigen::VectorXd::Constant(1, beta(2)+beta(1));
      case 3:
      return Eigen::VectorXd::Constant(1, 3.0*beta(2));
    }
    return Eigen::VectorXd::Zero(1);
  }

  /// Implement the pentalty function gradients
  /**
  The penalty functions ar
  \f{equation*}{
  \begin{array}{ccccc}
  f_0(\beta) = \beta_0, & f_1(\beta) = 1-\beta_1, & f_2(\beta) = \beta_2 + \beta_1, & \mbox{and} & f_3(\beta) = 3 \beta_2.
  \end{array}
  \f}
  @param[in] ind The index of the penalty function we are implementing
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$, each entry holds the index and value of a non-zero entry
  */
  inline virtual std::vector<std::pair<std::size_t, double> > PenaltyFunctionJacobianSparseImpl(std::size_t const ind) const override {
    switch( ind ) {
      case 0:
      return std::vector<std::pair<std::size_t, double> >({std::pair<std::size_t, double>(0, 1.0)});
      case 1:
      return std::vector<std::pair<std::size_t, double> >({std::pair<std::size_t, double>(1, -1.0)});
      case 2:
      return std::vector<std::pair<std::size_t, double> >({std::pair<std::size_t, double>(1, 1.0), std::pair<std::size_t, double>(2, 1.0)});
      case 3:
      return std::vector<std::pair<std::size_t, double> >({std::pair<std::size_t, double>(2, 3.0)});
    }
    return std::vector<std::pair<std::size_t, double> >();
  }

private:
};

} // namespace tests
} // namespace clf

#endif
