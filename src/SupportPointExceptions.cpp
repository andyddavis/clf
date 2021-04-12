#include "clf/SupportPointExceptions.hpp"

using namespace clf;

SupportPointBasisException::SupportPointBasisException(std::string const& basisType) : CLFException(), basisType(basisType) {
  message = "ERROR: SupportPoint tried to create the invalid basis type \"" + basisType + "\" (clf::SupportPointBasisException).";
}
