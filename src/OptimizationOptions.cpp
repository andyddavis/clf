#include "clf/OptimizationOptions.hpp"

namespace pt = boost::property_tree;
using namespace clf;

OptimizationOptions::OptimizationOptions(pt::ptree const& pt) :
useNLOPT(pt.get<bool>("UseNLOPT", false)),
useGaussNewtonHessian(pt.get<bool>("UseGaussNewtonHessian", true)),
maxEvals(pt.get<std::size_t>("MaxEvaluations", 250)),
algNLOPT(pt.get<std::string>("NLOPTAlgorithm", "LBFGS")),
atol_function(pt.get<double>("AbsoluteFunctionTol", 1.0e-10)),
rtol_function(pt.get<double>("RelativeFunctionTol", 1.0e-10)),
atol_step(pt.get<double>("AbsoluteStepSizeTol", 1.0e-10)),
rtol_step(pt.get<double>("RelativeStepSizeTol", 1.0e-10)),
atol_grad(pt.get<double>("AbsoluteGradientTol", 1.0e-10)),
maxStepSizeScale(pt.get<double>("MaxLineSearchStepSize", 1.0)),
maxLineSearchIterations(pt.get<std::size_t>("MaxLineSearchIterations", 10)),
lineSearchFactor(pt.get<double>("LineSearchFactor", 0.5))
{
  assert(atol_function>-1.0e-10); assert(rtol_function>-1.0e-10);
  assert(atol_step>-1.0e-10); assert(rtol_step>-1.0e-10);
  assert(atol_grad>-1.0e-10);
  assert(lineSearchFactor<1.0);
}
