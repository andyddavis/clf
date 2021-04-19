#include "clf/Model.hpp"

namespace pt = boost::property_tree;
using namespace clf;

Model::Model(pt::ptree const& pt) :
inputDimension(pt.get<std::size_t>("InputDimension", 1)),
outputDimension(pt.get<std::size_t>("OutputDimension", 1))
{}
