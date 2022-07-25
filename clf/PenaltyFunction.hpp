#ifndef PENALTYFUNCTION_HPP_
#define PENALTYFUNCTION_HPP_

#include <memory>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "clf/CLFExceptions.hpp"

#include "clf/Parameters.hpp"

namespace clf {

/// A penalty function \f$c: \mathbb{R}^{d} \mapsto \mathbb{R}^{n}\f$ for a clf::CostFunction
/**
   <B>Configuration Parameters:</B>
   Parameter Key | Type | Default Value | Description |
   ------------- | ------------- | ------------- | ------------- |
   "DeltaFD"   | <tt>double</tt> | <tt>1.0e-2</tt> | The step size for the finite difference approximation (see PenaltyFunction::deltaFD_DEFAULT). |
   "OrderFD"   | <tt>std::size_t</tt> | <tt>8</tt> | The accuracy order for the finite difference approximation (see PenaltyFunction::orderFD_DEFAULT). The options are \f$2\f$, \f$4\f$, \f$6\f$, and \f$8\f$. |
 */
template<typename MatrixType>
class PenaltyFunction {
public:

  /**
     @param[in] indim The input dimension \f$d\f$
     @param[in] outdim The output dimension \f$n\f$
     @param[in] para The parameters for this penalty function 
  */
  inline PenaltyFunction(std::size_t const indim, std::size_t const outdim, std::shared_ptr<const Parameters> const& para = std::make_shared<Parameters>()) :
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
     \return The Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$
   */
  virtual MatrixType JacobianFD(Eigen::VectorXd const& beta) = 0;

  /// Compute the weighted sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} w_i \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
  /**
     @param[in] beta The input parameters \f$\beta\f$
     @param[in] weights The weights for the weighted sum
     \return The weighted sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} w_i \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
   */
  inline virtual MatrixType Hessian(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) { return HessianFD(beta, weights); }

  /// Compute the weighted sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} w_i \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$ using finite difference
  /**
     @param[in] beta The input parameters \f$\beta\f$
     @param[in] weights The weights for the weighted sum
     \return The weighted sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} w_i \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
   */
  virtual MatrixType HessianFD(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) = 0;

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
  
  /// Compute the first derivative of \f$i^{\text{th}}\f$ component with respect to each input \f$\nabla_{\beta} c_i\f$ using finite difference
  /**
     @param[in] component We want the first derivative of the \f$i^{\th}\f$ component 
     @param[in] delta The step size for the finite difference 
     @param[in] weights The weights for the finite difference 
     @param[in,out] beta The input parameters, pass by reference to avoid having to copy to modify with the step size. Should not actually be changed at the end.
   */
  inline Eigen::VectorXd FirstDerivativeFD(std::size_t const component, double const delta, Eigen::VectorXd const& weights, Eigen::VectorXd& beta) {
    Eigen::VectorXd vec = Eigen::VectorXd::Zero(outdim);
    
    for( std::size_t j=0; j<weights.size(); ++j ) {
      beta(component) += delta;
      vec += weights(j)*Evaluate(beta);
    }
    beta(component) -= weights.size()*delta;
    for( std::size_t j=0; j<weights.size(); ++j ) {
      beta(component) -= delta;
      vec -= weights(j)*Evaluate(beta);
    }
    beta(component) += weights.size()*delta;

    return vec;
  }

  /// The parameters for this penalty function 
  /**
     Used to determine the parameters for the finite different approximation of the Jacobian and Hessian.
   */
  std::shared_ptr<const Parameters> para;

  /// The default value for the finite diference delta
  inline static double deltaFD_DEFAULT = 1.0e-2;

  /// The default value for the finite diference order
  inline static std::size_t orderFD_DEFAULT = 8;

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
  DensePenaltyFunction(std::size_t const indim, std::size_t const outdim, std::shared_ptr<const Parameters> const& para = std::make_shared<Parameters>());
  
  virtual ~DensePenaltyFunction() = default;
  
  /// Compute the Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$ using finite difference
  /**
     @param[in] beta The input parameters \f$\beta\f$
     \return The Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$
   */
  virtual Eigen::MatrixXd JacobianFD(Eigen::VectorXd const& beta) final override;

  /// Compute the weighted sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} w_i \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$ using finite difference
  /**
     @param[in] beta The input parameters \f$\beta\f$
     @param[in] weights The weights for the weighted sum
     \return The weighted sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} w_i \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
   */
  virtual Eigen::MatrixXd HessianFD(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) final override;
  
private:
};

/// A penalty function (see clf::PenaltyFunction) used to define a cost fuction (see clf::CostFunction) that can be minimized using the Levenberg Marquardt algorithm (see clf::LevenbergMarquardt) using sparse matrices
/**
   <B>Configuration Parameters:</B>

   In addition to the parameters required by clf::PenaltyFunction, the sparse version uses:
   Parameter Key | Type | Default Value | Description |
   ------------- | ------------- | ------------- | ------------- |
   "SparsityTolerance"   | <tt>double</tt> | <tt>1.0e-14</tt> | When adding to a sparse matrix, any value less than this tolerance is conserided zero (see SparsePenaltyFunction::sparsityTolerance_DEFAULT). |
*/
class SparsePenaltyFunction : public PenaltyFunction<Eigen::SparseMatrix<double> > {
public:
  
  /**
     @param[in] indim The input dimension \f$d\f$
     @param[in] outdim The output dimension \f$n\f$
     @param[in] para The parameters for this penalty function 
  */
  SparsePenaltyFunction(std::size_t const indim, std::size_t const outdim, std::shared_ptr<const Parameters> const& para = std::make_shared<Parameters>());
  
  virtual ~SparsePenaltyFunction() = default;

  /// Compute the Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$
  /**
     @param[in] beta The input parameters \f$\beta\f$
     \return The Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$\
   */
  virtual Eigen::SparseMatrix<double> Jacobian(Eigen::VectorXd const& beta) final override;

  /// Compute the Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$ using finite difference
  /**
     @param[in] beta The input parameters \f$\beta\f$
     \return The Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$\
  */
  virtual Eigen::SparseMatrix<double> JacobianFD(Eigen::VectorXd const& beta) final override;

  /// Compute the terms of the sparse Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$ using
  /**
     @param[in] beta The input parameters \f$\beta\f$
     @param[out] entries The entries of the Jacobian matrix
   */
  virtual void JacobianEntries(Eigen::VectorXd const& beta, std::vector<Eigen::Triplet<double> >& entries);

  /// Compute the terms of the sparse Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$ using finite difference
  /**
     @param[in] beta The input parameters \f$\beta\f$
     @param[out] entries The entries of the Jacobian matrix
   */
  void JacobianEntriesFD(Eigen::VectorXd const& beta, std::vector<Eigen::Triplet<double> >& entries);

  /// Compute entries of the weighted sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} w_i \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
  /**
     @param[in] beta The input parameters \f$\beta\f$
     @param[in] weights The weights for the weighted sum
     @param[out] entries The entries of the Hessian matrix
   */
  virtual void HessianEntries(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights, std::vector<Eigen::Triplet<double> >& entries);

  /// Compute entries of the weighted sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} w_i \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$ using finite difference
  /**
     @param[in] beta The input parameters \f$\beta\f$
     @param[in] weights The weights for the weighted sum
     @param[out] entries The entries of the Hessian matrix
   */
  void HessianEntriesFD(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights, std::vector<Eigen::Triplet<double> >& entries);

  /// Compute weighted the sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} w_i \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
  /**
     @param[in] beta The input parameters \f$\beta\f$
     @param[in] weights The weights for the weighted sum
     \return The weighted sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} w_i \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
   */
  virtual Eigen::SparseMatrix<double> Hessian(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) final override;

  /// Compute the weighted sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$ using finite difference
  /**
     @param[in] beta The input parameters \f$\beta\f$
     @param[in] weights The weights for the weighted sum
     \return The sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} w_i \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$
   */
  virtual Eigen::SparseMatrix<double> HessianFD(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights) final override;

private:

  /// The default value for the sparsity tolerance. 
  /**
     When adding to a sparse matrix, any value less than this tolerance is conserided zero.
  */
  inline static double sparsityTolerance_DEFAULT = 1.0e-14;
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
