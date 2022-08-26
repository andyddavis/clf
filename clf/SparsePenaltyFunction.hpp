#ifndef SPARSEPENALTYFUNCTION_HPP_
#define SPARSEPENALTYFUNCTION_HPP_

#include "clf/PenaltyFunction.hpp"

namespace clf {

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
     \return The entries of the Jacobian matrix
   */
  virtual std::vector<Eigen::Triplet<double> > JacobianEntries(Eigen::VectorXd const& beta);

  /// Compute the terms of the sparse Jacobian of the penalty function \f$\nabla_{\beta} c \in \mathbb{R}^{n \times d}\f$ using finite difference
  /**
     @param[in] beta The input parameters \f$\beta\f$
     \return The entries of the Jacobian matrix
   */
  std::vector<Eigen::Triplet<double> > JacobianEntriesFD(Eigen::VectorXd const& beta);

  /// Compute entries of the weighted sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} w_i \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$ 
  /**
     @param[in] beta The input parameters \f$\beta\f$
     @param[in] weights The weights for the weighted sum
     \return The entries of the Hessian matrix
   */
  virtual std::vector<Eigen::Triplet<double> > HessianEntries(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights);

  /// Compute entries of the weighted sum of the Hessian of each the penalty function \f$\sum_{i=1}^{n} w_i \nabla_{\beta}^2 c_i \in \mathbb{R}^{d \times d}\f$ using finite difference
  /**
     @param[in] beta The input parameters \f$\beta\f$
     @param[in] weights The weights for the weighted sum
     \return The entries of the Hessian matrix
   */
  std::vector<Eigen::Triplet<double> > HessianEntriesFD(Eigen::VectorXd const& beta, Eigen::VectorXd const& weights);

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

} // namespace clf

#endif
