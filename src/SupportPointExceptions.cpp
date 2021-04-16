#include "clf/SupportPointExceptions.hpp"

using namespace clf;

const std::vector<std::string> SupportPointBasisException::options =
    {
      "TOTALORDERPOLYNOMIALS",
      "TOTALORDERSINCOS"
    };

SupportPointBasisException::SupportPointBasisException(std::string const& basisType) : CLFException(), basisType(basisType) {
  message = "ERROR: SupportPoint tried to create the invalid basis type \"" + basisType + "\", options are:\n";
  for( const auto& it : options ) { message += it + "\n"; }
  message += "(clf::SupportPointBasisException).";
}
