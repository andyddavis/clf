#ifndef COSTFUNCTION_HPP_
#define COSTFUNCTION_HPP_

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace clf {

/// A cost function that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt)
template<typename MatrixType>
class CostFunction {
public:

  /**
     @param[in] indim The number of parameters for the cost function
   */
  inline CostFunction(std::size_t const indim) :
    indim(indim)
  {}

  virtual ~CostFunction() = default;

  /// The number of parameters for the cost function
  const std::size_t indim;

private:
};

/// A cost function (see clf::CostFunction) that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt) using dense matrices
typedef CostFunction<Eigen::MatrixXd> DenseCostFunction;

/// A cost function (see clf::CostFunction) that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt) using sparse matrices
typedef CostFunction<Eigen::SparseMatrix<double> > SparseCostFunction;

} // namespace clf 

#endif
