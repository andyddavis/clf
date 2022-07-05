#ifndef LEVENBERGMARQUARDT_HPP_
#define LEVENBERGMARQUARDT_HPP_

#include "clf/Parameters.hpp"
#include "clf/CostFunction.hpp"

namespace clf {

namespace Optimization {

/// Information about whether or not the algorithm converged
enum Convergence {
  /// Failed for some unknown or unspecified reason
  FAILED = -1,

  /// The algorithm is currently running
  CONTINUE_RUNNING = 0,
};

} // namespace Optimization

/// An implementation of the Levenberg Marquardt algorithm 
template<typename MatrixType>
class LevenbergMarquardt {
public:

  /**
     @param[in] para The parameters for this algorithm
   */
  inline LevenbergMarquardt(std::shared_ptr<Parameters> const& para) {}

  virtual ~LevenbergMarquardt() = default;

  /// Minimize the cost function using the Levenberg Marquardt algorithm
  /**
     @param[in, out] beta In: The initial guess for the Levenberg Marquardt algorithm; Out: The
     paramter values that minimizes the cost function
     @param[out] costVec The evaluation of the cost function at the optimal point
     \return First: Information about convergence or failure , Second: The current cost
  */
  inline std::pair<Optimization::Convergence, double> Minimize(Eigen::VectorXd& beta, Eigen::VectorXd& costVec) {
    return std::pair<Optimization::Convergence, double>(Optimization::Convergence::FAILED, std::numeric_limits<double>::quiet_NaN());
  }
  
private:
};

/// The Levenverg Marquardt (see clf::LevenbergMarquardt) optimization algorithm using dense matrices
typedef LevenbergMarquardt<Eigen::MatrixXd> DenseLevenbergMarquardt;

/// The Levenverg Marquardt (see clf::LevenbergMarquardt) optimization algorithm using sparse matrices
typedef LevenbergMarquardt<Eigen::SparseMatrix<double> > SparseLevenbergMarquardt;

} // namespace clf

#endif
