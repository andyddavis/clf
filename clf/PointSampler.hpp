#ifndef POINTSAMPLER_HPP_
#define POINTSAMPLER_HPP_

#include <MUQ/Modeling/Distributions/RandomVariable.h>

#include "clf/Model.hpp"

namespace clf {

/// Generate random points from a given distribution
class PointSampler {
public:

  /**
  @param[in] randVar This allows us to sample from the distribution \f$\pi\f$
  @param[in] model The clf::Model associated with each support point
  */
  PointSampler(std::shared_ptr<muq::Modeling::RandomVariable> const& randVar, std::shared_ptr<Model> const& model);

  virtual ~PointSampler() = default;

protected:

  /// Sample a location \f$x \sim \pi\f$
  /**
  \return The sampled location \f$x\f$
  */
  Eigen::VectorXd SampleLocation() const;

  /// The clf::Model associated with each support point
  std::shared_ptr<Model> model;

private:

  /// This allows us to sample from the distribution \f$\pi\f$
  std::shared_ptr<muq::Modeling::RandomVariable> randVar;

};

} // namespace clf

#endif
