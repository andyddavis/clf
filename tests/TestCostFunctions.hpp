#ifndef TESTCOSTFUNCTIONS_HPP_
#define TESTCOSTFUNCTIONS_HPP_

#include "clf/CostFunction.hpp"

namespace clf {
namespace tests {

/// An example cost function used to test clf::DenseCostFunction
/**
The input dimension is \f$3\f$ and the number of penalty functions is \f$4\f$. The penalty functions are
\f{equation*}{
\begin{array}{ccccc}
f_0(\beta) = \beta_0, & f_1(\beta) = 1-\beta_1, & f_2(\beta) = \beta_2, & \mbox{and} & f_3(\beta) = \beta_0 \beta_2.
\end{array}
\f}
*/
class DenseCostTest : public DenseCostFunction {
public:

  /// The input dimension is \f$3\f$ and the number of penalty terms is \f$4\f$
  inline DenseCostTest() : DenseCostFunction(3, 4) {}

  virtual ~DenseCostTest() = default;

protected:

  /// Implement the pentalty functions
  /**
  The penalty functions are
  \f{equation*}{
  \begin{array}{ccccc}
  f_0(\beta) = \beta_0, & f_1(\beta) = 1-\beta_1, & f_2(\beta) = \beta_2, & \mbox{and} & f_3(\beta) = \beta_0 \beta_2.
  \end{array}
  \f}
  @param[in] ind The index of the penalty function we are implementing
  @param[in] beta The parameter value
  \return The \f$i^{th}\f$ penalty function \f$f_i(\beta)\f$
  */
  inline virtual double PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const override {
    switch( ind ) {
      case 0:
      return beta(0);
      case 1:
      return 1.0-beta(1);
      case 2:
      return beta(2);
      case 3:
      return beta(0)*beta(2);
    }
    return 0.0;
  }

  /// Implement the pentalty function gradients
  /**
  The penalty functions are
  \f{equation*}{
  \begin{array}{ccccc}
  f_0(\beta) = \beta_0, & f_1(\beta) = 1-\beta_1, & f_2(\beta) = \beta_2, & \mbox{and} & f_3(\beta) = \beta_0 \beta_2.
  \end{array}
  \f}
  @param[in] ind The index of the penalty function we are implementing
  @param[in] beta The parameter value
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$
  */
  inline virtual Eigen::VectorXd PenaltyFunctionGradientImpl(std::size_t const ind, Eigen::VectorXd const& beta) const override {
    switch( ind ) {
      case 0:
      return Eigen::Vector3d(1.0, 0.0, 0.0);
      case 1:
      return Eigen::Vector3d(0.0, -1.0, 0.0);
      case 2:
      return Eigen::Vector3d(0.0, 0.0, 1.0);
      case 3:
      return Eigen::Vector3d(beta(2), 0.0, beta(0));
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
f_0(\beta) = \beta_0, & f_1(\beta) = 1-\beta_1, & f_2(\beta) = \beta_2, & \mbox{and} & f_3(\beta) = \beta_0 \beta_2.
\end{array}
\f}
*/
class SparseCostTest : public SparseCostFunction {
public:

  /// The input dimension is \f$3\f$ and the number of penalty terms is \f$4\f$
  inline SparseCostTest() : SparseCostFunction(3, 4) {}

  virtual ~SparseCostTest() = default;

protected:

  /// Implement the pentalty functions
  /**
  The penalty functions are
  \f{equation*}{
  \begin{array}{ccccc}
  f_0(\beta) = \beta_0, & f_1(\beta) = 1-\beta_1, & f_2(\beta) = \beta_2, & \mbox{and} & f_3(\beta) = \beta_0 \beta_2.
  \end{array}
  \f}
  @param[in] ind The index of the penalty function we are implementing
  @param[in] beta The parameter value
  \return The \f$i^{th}\f$ penalty function \f$f_i(\beta)\f$
  */
  inline virtual double PenaltyFunctionImpl(std::size_t const ind, Eigen::VectorXd const& beta) const override {
    switch( ind ) {
      case 0:
      return beta(0);
      case 1:
      return 1.0-beta(1);
      case 2:
      return beta(2);
      case 3:
      return beta(0)*beta(2);
    }
    return 0.0;
  }

  /// Implement the pentalty function gradients
  /**
  The penalty functions are
  \f{equation*}{
  \begin{array}{ccccc}
  f_0(\beta) = \beta_0, & f_1(\beta) = 1-\beta_1, & f_2(\beta) = \beta_2, & \mbox{and} & f_3(\beta) = \beta_0 \beta_2.
  \end{array}
  \f}
  @param[in] ind The index of the penalty function we are implementing
  @param[in] beta The parameter value
  \return The gradient of the \f$i^{th}\f$ penalty function \f$\nabla_{\beta} f_i(\beta)\f$, each entry holds the index and value of a non-zero entry
  */
  inline virtual std::vector<std::pair<std::size_t, double> > PenaltyFunctionGradientSparseImpl(std::size_t const ind, Eigen::VectorXd const& beta) const override {
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

} // namespace tests
} // namespace clf

#endif
