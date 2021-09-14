#ifndef COLOCATIONPOINTSAMPLER_HPP_
#define COLOCATIONPOINTSAMPLER_HPP_

#include <MUQ/Modeling/Distributions/RandomVariable.h>

#include "clf/ColocationPoint.hpp"

namespace clf {

/// Generate colocation points from the colocation distribution
class ColocationPointSampler {
public:

  /**
  @param[in] randVar The distribution that we use to sample the point location
  @param[in] model The model associated with each point
  */
  ColocationPointSampler(std::shared_ptr<muq::Modeling::RandomVariable> const& randVar, std::shared_ptr<Model> const& model);

  virtual ~ColocationPointSampler() = default;

  /// Sample a colocation point
  /**
  \return A sampled colocation point
  */
  std::shared_ptr<ColocationPoint> Sample() const;

  /// The input dimension
  std::size_t InputDimension() const;

  /// The output dimension
  std::size_t OutputDimension() const;

private:

  /// The distribution we use to sample colocation point locations
  std::shared_ptr<muq::Modeling::RandomVariable> randVar;

  /// The model that we are trying to solve
  std::shared_ptr<Model> model;
};

} // namespace clf

#endif
