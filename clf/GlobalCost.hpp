#ifndef GLOBALCOST_HPP_
#define GLOBALCOST_HPP_

#include <boost/property_tree/ptree.hpp>

#include <Eigen/Sparse>

#include <MUQ/Optimization/CostFunction.h>

#include "clf/DOFIndices.hpp"

namespace clf {

/// Compute the global cost of the coefficients for <em>all</em> of the support points in a clf::SupportPointCloud
class GlobalCost : public muq::Optimization::CostFunction {
public:

  /**
  @param[in] cloud The support point cloud that holds all of the points (and their local cost functions)
  @param[in] pt The options for the global cost function
  */
  GlobalCost(std::shared_ptr<SupportPointCloud> const& cloud, boost::property_tree::ptree const& pt);

  virtual ~GlobalCost() = default;

  // Compute the gradient of the cost given the coefficients
  /**
  @param[in] coefficients The coefficients for the basis functions
  \return The cost gradient
  */
  Eigen::VectorXd Gradient(Eigen::VectorXd const& coefficients) const;

  /// Compute the Hessian
  /**
  @param[in] coefficients The basis function coefficients for <em>all</em> of the support points
  @param[in] gaussNewtonHessian <tt>true</tt>: compute the Gauss-Newton Hessian, <tt>false</tt>: compute the Hessian
  @param[out] hess The Hessian matrix
  */
  void Hessian(Eigen::VectorXd const& coefficients, bool const gaussNewtonHessian, Eigen::SparseMatrix<double>& hess) const;

protected:

  /// Compute the cost function
  /**
  @param[in] input There is only one input: the basis function coefficients for <em>all</em> of the support points
  */
  virtual double CostImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& input) override;

  /// Compute the cost function gradient
  /**
  @param[in] inputDimWrt Since there is only one input, this should always be zero
  @param[in] input There is only one input: the basis function coefficients for <em>all</em> of the support points
  @param[in] sensitivity A scaling for the gradient
  */
  virtual void GradientImpl(unsigned int const inputDimWrt, muq::Modeling::ref_vector<Eigen::VectorXd> const& input, Eigen::VectorXd const& sensitivity) override;

private:

  /// The map from support point indices to the global degree of freedom indices
  const DOFIndices dofIndices;
};

} // namespace clf

#endif
