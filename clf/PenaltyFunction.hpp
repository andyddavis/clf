#ifndef PENALTYFUNCTION_HPP_
#define PENALTYFUNCTION_HPP_

#include <memory>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "clf/Parameters.hpp"

namespace clf {

/// A penalty function \f$c: \mathbb{R}^{d} \mapsto \mathbb{R}^{n}\f$ for a clf::CostFunction
/**
   <B>Configuration Parameters:</B>
   Parameter Key | Type | Default Value | Description |
   ------------- | ------------- | ------------- | ------------- |
   "DeltaJacobain"   | <tt>double</tt> | <tt>1.0e-2</tt> | The step size for the finite difference approximation of the Jacobian. |
 */
template<typename MatrixType>
class PenaltyFunction {
public:

  /**
     @param[in] indim The input dimension \f$d\f$
     @param[in] outdim The output dimension \f$n\f$
     @param[in] para The parameters for this penalty function 
  */
  inline PenaltyFunction(std::size_t const indim, std::size_t const outdim, std::shared_ptr<Parameters> const& para = std::make_shared<Parameters>()) :
    indim(indim), outdim(outdim), para(para)
  {}

  virtual ~PenaltyFunction() = default;

  /// Evaluate the penalty function \f$c: \mathbb{R}^{d} \mapsto \mathbb{R}^{n}\f$
  /**
     @param[in] beta The input parameters \f$\beta\f$
     \return The penalty function evaluation \f$c(\beta)\f$
   */
  virtual Eigen::VectorXd Evaluate(Eigen::VectorXd const& beta) = 0;

  /// Compute the Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$
  /**
     @param[in] beta The input parameters \f$\beta\f$
     \return The Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$\
   */
  inline virtual MatrixType Jacobian(Eigen::VectorXd const& beta) { return JacobianFD(beta); }

  /// Compute the Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$ using finite difference
  /**
     @param[in] beta The input parameters \f$\beta\f$
     \return The Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$\
   */
  virtual MatrixType JacobianFD(Eigen::VectorXd beta) = 0;

  /// The input dimension \f$d\f$
  const std::size_t indim;

  /// The output dimension \f$n\f$
  const std::size_t outdim;

protected:

  /// Get the weights used for the finite difference approximation
  /**
     @param[in] order The accuracy order for the finite difference approximation
     \return The weights using centered difference
   */
  inline Eigen::VectorXd FiniteDifferenceWeights(std::size_t const order) const {
    Eigen::VectorXd weights;
    switch( order ) { 
    case 2: 
      weights.resize(1);
      weights << 0.5;
      return weights;
    case 4: 
      weights.resize(2);
      weights << 2.0/3.0, -1.0/12.0;
      return weights;
    case 6: 
      weights.resize(3);
      weights << 0.75, -3.0/20.0, 1.0/60;
      return weights;
    default: // default to 8th order
      weights.resize(4);
      weights << 0.8, -0.2, 4.0/105.0, -1.0/280.0;
      return weights;
    }
  }
  
  /// The parameters for this penalty function 
  /**
     Used to determine the parameters for the finite different approximation of the Jacobian and Hessian.
   */
  std::shared_ptr<const Parameters> para;

private:

};

/// A penalty function (see clf::PenaltyFunction) used to define a cost fuction (see clf::CostFunction) that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt) using dense matrices
class DensePenaltyFunction : public PenaltyFunction<Eigen::MatrixXd> {
public:

  /**
     @param[in] indim The input dimension \f$d\f$
     @param[in] outdim The output dimension \f$n\f$
     @param[in] para The parameters for this penalty function 
  */
  DensePenaltyFunction(std::size_t const indim, std::size_t const outdim, std::shared_ptr<Parameters> const& para = std::make_shared<Parameters>());
  
  virtual ~DensePenaltyFunction() = default;
  
  /// Compute the Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$ using finite difference
  /**
     @param[in] beta The input parameters \f$\beta\f$
     \return The Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$\
   */
  virtual Eigen::MatrixXd JacobianFD(Eigen::VectorXd beta) final override;
  
private:
};

/// A penalty function (see clf::PenaltyFunction) used to define a cost fuction (see clf::CostFunction) that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt) using sparse matrices
class SparsePenaltyFunction : public PenaltyFunction<Eigen::SparseMatrix<double> > {
public:
  
  /**
     @param[in] indim The input dimension \f$d\f$
     @param[in] outdim The output dimension \f$n\f$
     @param[in] para The parameters for this penalty function 
  */
  SparsePenaltyFunction(std::size_t const indim, std::size_t const outdim, std::shared_ptr<Parameters> const& para = std::make_shared<Parameters>());
  
  virtual ~SparsePenaltyFunction() = default;

  /// Compute the Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$ using finite difference
  /**
     @param[in] beta The input parameters \f$\beta\f$
     \return The Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$\
   */
  inline virtual Eigen::SparseMatrix<double> JacobianFD(Eigen::VectorXd beta) final override;

private:
};

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
