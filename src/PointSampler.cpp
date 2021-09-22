#include "clf/PointSampler.hpp"

using namespace muq::Modeling;
using namespace clf;

PointSampler::PointSampler(std::shared_ptr<RandomVariable> const& randVar, std::shared_ptr<Model> const& model) :
randVar(randVar),
model(model)
{}

Eigen::VectorXd PointSampler::SampleLocation() const { return randVar->Sample(); }
