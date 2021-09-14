#include "clf/ColocationPointSampler.hpp"

using namespace muq::Modeling;
using namespace clf;

ColocationPointSampler::ColocationPointSampler(std::shared_ptr<RandomVariable> const& randVar, std::shared_ptr<Model> const& model) :
randVar(randVar),
model(model)
{}

std::shared_ptr<ColocationPoint> ColocationPointSampler::Sample() const { return std::make_shared<ColocationPoint>(randVar->Sample(), model); }

std::size_t ColocationPointSampler::InputDimension() const { return model->inputDimension; }

std::size_t ColocationPointSampler::OutputDimension() const { return model->outputDimension; }
