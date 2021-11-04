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
  By default, simply sample the location using clf::PointSampler::SampleLocation and create a collocation point with weight \f$n_c^{-1}\f$
  @param[in] ind The global index of the collocation point
  @param[in] num The total number of collocation points in the collocation point cloud (the number \f$n_c\f$)
  \return A sampled collocation point
  */
  virtual std::shared_ptr<CollocationPoint> Sample(std::size_t const ind, std::size_t const num) const;

private:
};

} // namespace clf

#endif
