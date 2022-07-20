#include "clf/CLFExceptions.hpp"

using namespace clf::exceptions;

NotImplemented::NotImplemented(std::string const& name) : std::logic_error("CLF Error: " + name + " not yet implemented") {}

