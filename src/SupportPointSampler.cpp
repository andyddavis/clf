#include "clf/SupportPointSampler.hpp"

#include "clf/LinearModel.hpp"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace clf;

SupportPointSampler::SupportPointSampler(std::shared_ptr<RandomVariable> const& randVar, std::shared_ptr<Model> const& model, pt::ptree const& options) :
PointSampler(randVar, model),
options(options.get_child("SupportPoint"))
{}

SupportPointSampler::SupportPointSampler(std::shared_ptr<muq::Modeling::RandomVariable> const& randVar, pt::ptree const& options) :
PointSampler(randVar, std::make_shared<LinearModel>(randVar->varSize, options.get<std::size_t>("OutputDimension", 1))),
options(options.get_child("SupportPoint"))
{}

std::shared_ptr<SupportPoint> SupportPointSampler::Sample() const { return SupportPoint::Construct(SampleLocation(), model, options); }
