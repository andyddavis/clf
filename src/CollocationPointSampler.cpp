#include "clf/CollocationPointSampler.hpp"

using namespace muq::Modeling;
using namespace clf;

CollocationPointSampler::CollocationPointSampler(std::shared_ptr<RandomVariable> const& randVar, std::shared_ptr<Model> const& model) :
PointSampler(randVar, model)
{}

std::shared_ptr<CollocationPoint> CollocationPointSampler::Sample() const { return std::make_shared<CollocationPoint>(SampleLocation(), model); }
