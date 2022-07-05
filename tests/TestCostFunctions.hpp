#ifndef TESTCOSTFUNCTIONS_HPP_
#define TESTCOSTFUNCTIONS_HPP_

#include "clf/CostFunction.hpp"

namespace clf {
namespace tests {

/// An example cost function used to test clf::DenseCostFunction
class DenseCostFunctionTest : public DenseCostFunction {
public:

  /**
     @param[in] indim The number of parameters for the cost function
   */
  inline DenseCostFunctionTest(std::size_t const indim) :
    DenseCostFunction(indim)
  {}

  virtual ~DenseCostFunctionTest() = default;

private:
};

/// An example cost function used to test clf::SparseCostFunction
class SparseCostFunctionTest : public SparseCostFunction {
public:

  /**
     @param[in] indim The number of parameters for the cost function
   */
  inline SparseCostFunctionTest(std::size_t const indim) :
    SparseCostFunction(indim)
  {}

  virtual ~SparseCostFunctionTest() = default;

private:
};

} // namespace tests
} // namespace clf 

#endif
