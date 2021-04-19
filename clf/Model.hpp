#ifndef MODEL_HPP_
#define MODEL_HPP_

#include <iostream>

#include <boost/property_tree/ptree.hpp>

#include <Eigen/Core>

#include "clf/ModelExceptions.hpp"

namespace clf {

/// Implements a model that is given to a clf::SupportPoint
/**
Each support point is associated with a model
\f{equation*}{
\mathcal{L}(u) = f,
\f}
where \f$u:\Omega \mapsto \mathbb{R}^{m}\f$ (and \f$\Omega \subseteq \mathbb{R}^{d}\f$) is the function that we are trying to represent with local functions. This class allows the user to implement \f$\mathcal{L}\f$ and/or \f$f\f$. By default, we assume \f$\mathcal{L}\f$ is the identity but the user <em>must</em> implement \f$f\f$.

<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"InputDimension"   | <tt>std::size_t</tt> | <tt>1</tt> | The input dimension \f$d\f$. |
"OutputDimension"   | <tt>std::size_t</tt> | <tt>1</tt> | The output dimension \f$m\f$. |
*/
class Model {
public:

  /**
  @param[in] pt Options for the model
  */
  Model(boost::property_tree::ptree const& pt);

  virtual ~Model() = default;

  /// Implement the right hand side function \f$f\f$
  /**
  @param[in] x The point \f$x \in \Omega \f$
  \return The evaluation of \f$f(x)\f$
  */
  Eigen::VectorXd RightHandSide(Eigen::VectorXd const& x) const;

  /// The input dimension \f$d\f$
  const std::size_t inputDimension;

  /// The output dimension \f$m\f$
  const std::size_t outputDimension;

protected:

  /// Implement the right hand side function \f$f\f$
  /**
  @param[in] x The point \f$x \in \Omega \f$
  \return The evaluation of \f$f(x)\f$
  */
  virtual Eigen::VectorXd RightHandSideVectorImpl(Eigen::VectorXd const& x) const;

  /// Implement the right hand side function \f$f\f$
  /**
  @param[in] x The point \f$x \in \Omega \f$
  @param[in] outind Return this component of the evaluation of \f$f\f$
  \return The component of \f$f(x)\f$ corresponding to <tt>outind</tt>
  */
  virtual double RightHandSideComponentImpl(Eigen::VectorXd const& x, std::size_t const outind) const;

private:
};

} // namespace clf

#endif
