#ifndef COLLOCATIONPOINTSAMPLER_HPP_
#define COLLOCATIONPOINTSAMPLER_HPP_

#include <MUQ/Modeling/Distributions/RandomVariable.h>

#include "clf/CollocationPoint.hpp"
#include "clf/PointSampler.hpp"

namespace clf {

/// Generate collocation points from the colocation distribution
class CollocationPointSampler : public PointSampler {
public:

  /**
  @param[in] randVar The distribution that we use to sample the point location
  @param[in] model The model associated with each point
  */
  CollocationPointSampler(std::shared_ptr<muq::Modeling::RandomVariable> const& randVar, std::shared_ptr<Model> const& model);

  virtual ~CollocationPointSampler() = default;

  /// Sample a collocation point
  /**
  \return A sampled collocation point
  */
  std::shared_ptr<CollocationPoint> Sample() const;

private:
};

} // namespace clf

#endif
