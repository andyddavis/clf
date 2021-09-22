#include "clf/SupportPointSampler.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace clf;

SupportPointSampler::SupportPointSampler(std::shared_ptr<RandomVariable> const& randVar, std::shared_ptr<Model> const& model, pt::ptree const& options) :
PointSampler(randVar, model),
options(options.get_child("SupportPoint"))
{}

std::shared_ptr<SupportPoint> SupportPointSampler::Sample() const { return SupportPoint::Construct(SampleLocation(), model, options); }
