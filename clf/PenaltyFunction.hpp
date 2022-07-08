#ifndef PENALTYFUNCTION_HPP_
#define PENALTYFUNCTION_HPP_

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace clf {

/// A penalty function \f$c: \mathbb{R}^{d} \mapsto \mathbb{R}^{n}\f$ for a clf::CostFunction
template<typename MatrixType>
class PenaltyFunction {
public:

  /**
     @param[in] indim The input dimension \f$d\f$
     @param[in] outdim The output dimension \f$n\f$
  */
  inline PenaltyFunction(std::size_t const indim, std::size_t const outdim) :
    indim(indim), outdim(outdim)
  {}

  virtual ~PenaltyFunction() = default;

  /// Evaluate the penalty function \f$c: \mathbb{R}^{d} \mapsto \mathbb{R}^{n}\f$
  /**
     @param[in] beta The input parameters \f$\beta\f$
     \return The penalty function evaluation \f$c(\beta)\f$
   */
  virtual Eigen::VectorXd Evaluate(Eigen::VectorXd const& beta) = 0;

  /// The input dimension \f$d\f$
  const std::size_t indim;

  /// The output dimension \f$n\f$
  const std::size_t outdim;

private:
};

/// A penalty function (see clf::PenaltyFunction) used to define a cost fuction (see clf::CostFunction) that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt) using dense matrices
typedef PenaltyFunction<Eigen::MatrixXd> DensePenaltyFunction;

/// A penalty function (see clf::PenaltyFunction) used to define a cost fuction (see clf::CostFunction) that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt) using sparse matrices
typedef PenaltyFunction<Eigen::SparseMatrix<double> > SparsePenaltyFunction;

/// A vector of penalty functions (see clf::PenaltyFunction) used to define a cost fuction (see clf::CostFunction) that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt)
template<typename MatrixType>
using PenaltyFunctions = std::vector<std::shared_ptr<PenaltyFunction<MatrixType> > >;

/// A vector of constant penalty functions (see clf::PenaltyFunction) used to define a cost fuction (see clf::CostFunction) that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt)
template<typename MatrixType>
using ConstPenaltyFunctions = std::vector<std::shared_ptr<const PenaltyFunction<MatrixType> > >;

/// A vector of constant penalty functions (see clf::PenaltyFunction) used to define a cost fuction (see clf::CostFunction) that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt) using dense matrices
typedef ConstPenaltyFunctions<Eigen::MatrixXd> DenseConstPenaltyFunctions;

/// A vector of penalty functions (see clf::PenaltyFunction) used to define a cost fuction (see clf::CostFunction) that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt) using dense matrices
typedef PenaltyFunctions<Eigen::MatrixXd> DensePenaltyFunctions;

/// A vector of constant penalty functions (see clf::PenaltyFunction) used to define a cost fuction (see clf::CostFunction) that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt) using sparse matrices
typedef ConstPenaltyFunctions<Eigen::SparseMatrix<double> > SparseConstPenaltyFunctions;

/// A vector of penalty functions (see clf::PenaltyFunction) used to define a cost fuction (see clf::CostFunction) that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt) using sparse matrices
typedef PenaltyFunctions<Eigen::SparseMatrix<double> > SparsePenaltyFunctions;

} // namespace clf

#endif
