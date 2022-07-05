#include "clf/Parameters.hpp"

using namespace clf;

Parameters::Parameters() {}

std::size_t Parameters::NumParameters() const { return map.size(); }
