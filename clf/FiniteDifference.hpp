#ifndef FINITEDIFFERENCE_HPP_
#define FINITEDIFFERENCE_HPP_

#include <Eigen/Core>

namespace clf {
namespace FiniteDifference { 

/// Get the weights used for the finite difference approximation
/**
   @param[in] order The accuracy order for the finite difference approximation (options are \f$2\f$, \f$4\f$, \f$6\f$, and \f$8\f$, although the default is \f$8\f$ is an invalid order is given)
   \return The weights using centered difference
*/
Eigen::VectorXd Weights(std::size_t const order);

  /// Compute the derivative of \f$i^{\text{th}}\f$ component with respect to each input \f$\nabla_{\beta} c_i\f$ using finite difference
  /**
     @param[in] component We want the first derivative of the \f$i^{\th}\f$ component 
     @param[in] delta The step size for the finite difference 
     @param[in] weights The weights for the finite difference 
     @param[in,out] beta The input parameters, pass by reference to avoid having to copy to modify with the step size. Should not actually be changed at the end.
     @param[in] func Compute the derivative of this function 
     \return The derivative with respect to the \f$i^{\text{th}}\f$ component
   */
  template<typename TYPE>
  TYPE Derivative(std::size_t const component, double const delta, Eigen::VectorXd const& weights, Eigen::VectorXd& beta, std::function<TYPE(Eigen::VectorXd const&)> const& func) {
    beta(component) += delta;
    
    TYPE vec = weights(0)*func(beta);
    for( std::size_t j=1; j<weights.size(); ++j ) {
      beta(component) += delta;
      vec += weights(j)*func(beta);
    }
    beta(component) -= weights.size()*delta;
    for( std::size_t j=0; j<weights.size(); ++j ) {
      beta(component) -= delta;
      vec -= weights(j)*func(beta);
    }
    beta(component) += weights.size()*delta;
    
    return vec/delta;
  }

} // namespace FiniteDifference
} // namespace clf

#endif
