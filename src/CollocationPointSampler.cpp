#include "clf/CollocationPointSampler.hpp"

using namespace muq::Modeling;
using namespace clf;

CollocationPointSampler::CollocationPointSampler(std::shared_ptr<RandomVariable> const& randVar, std::shared_ptr<Model> const& model) :
PointSampler(randVar, model)
{}

std::shared_ptr<CollocationPoint> CollocationPointSampler::Sample(std::size_t const ind, std::size_t const num) const { return std::make_shared<CollocationPoint>(1.0/num, SampleLocation(), model); }
