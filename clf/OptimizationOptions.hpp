#ifndef OPTIMIZATIONOPTIONS_HPP_
#define OPTIMIZATIONOPTIONS_HPP_

#include <boost/property_tree/ptree.hpp>

namespace clf {

/// Options for the optimizers---stored in their own object for better organization
/**
<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"UseNLOPT"   | <tt>bool</tt> | <tt>false</tt> | Should we use <tt>NLOPT</tt> (<tt>true</tt>) or Newton's method using the Hessian (or ) Gauss-Newton Hessian (<tt>false</tt>)? |
"UseGaussNewtonHessian"   | <tt>bool</tt> | <tt>true</tt> | Should we use the Gauss-Newton Hessian or the true Hessian? |
"NLOPTAlgorithm"   | <tt>std::string</tt> | <tt>"LBFGS"</tt> | The <tt>NLOPT</tt> algorithm used for the optimization. |
"MaxEvaluations"   | <tt>std::size_t</tt> | <tt>250</tt> | The maximum number of evaluations. |
"AbsoluteFunctionTol"   | <tt>double</tt> | <tt>1.0e-10</tt> | The absolute cost function tolerance. |
"RelativeFunctionTol"   | <tt>double</tt> | <tt>1.0e-10</tt> | The relative cost function tolerance. |
"AbsoluteStepSizeTol"   | <tt>double</tt> | <tt>1.0e-10</tt> | The absolute step size tolerance. |
"RelativeStepSizeTol"   | <tt>double</tt> | <tt>1.0e-10</tt> | The relative step size tolerance. |
"AbsoluteGradientTol"   | <tt>double</tt> | <tt>1.0e-10</tt> | The absolute gradient tolerance. |
"MaxLineSearchStepSize"   | <tt>double</tt> | <tt>1.0</tt> | The maximum step size scale. |
"MaxLineSearchIterations"   | <tt>std::size_t</tt> | <tt>10</tt> | The maximum number of line search iterations. |
"LineSearchFactor"   | <tt>double</tt> | <tt>0.5</tt> | The amount the line search decreases the step size at each iteration. |
*/
struct OptimizationOptions {

  /**
  @param[in] pt The optimization options
  */
  OptimizationOptions(boost::property_tree::ptree const& pt);

  virtual ~OptimizationOptions() = default;

  /// Should we use <tt>NLOPT</tt> (<tt>true</tt>) or Newton's method using the Hessian (or ) Gauss-Newton Hessian (<tt>false</tt>)?
  const bool useNLOPT;

  /// Use the Gauss-Newton Hessian or the true Hessian?
  /**
  Only relavant for Newton's method.
  */
  const bool useGaussNewtonHessian;

  /// The maximum number of evaluations.
  /**
  In the Newton's method case this is actually the maximum number of iterations.
  */
  const std::size_t maxEvals;

  /// The <tt>NLOPT</tt> algorithm used for the optimization.
  /**
  Only used if useNLOPT is <tt>true</tt>.
  */
  const std::string algNLOPT;

  /// The absolute cost function tolerance.
  const double atol_function;

  /// The relative cost function tolerance.
  /**
  Relative to the initial cost.
  */
  const double rtol_function;

  /// The absolute step size tolerance.
  const double atol_step;

  /// The relative step size tolerance.
  /**
  The stepsize relative to the norm of the coefficients.
  */
  const double rtol_step;

  /// The absolute gradient tolerance (Newton's method only).
  const double atol_grad;

  ///Maximum step for the line search (Newton's method only).
  const double maxStepSizeScale;

  /// Maximum number of line search iterations (Newton's method only).
  const std::size_t maxLineSearchIterations;

  /// The amount the line search decreases the step size at each iteration (Newton's method only).
  const double lineSearchFactor;
};

} // namespace clf

#endif
