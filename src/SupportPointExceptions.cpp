#include "clf/SupportPointExceptions.hpp"

using namespace clf::exceptions;

SupportPointWrongNumberOfBasesConstructed::SupportPointWrongNumberOfBasesConstructed(std::size_t const outdim, std::size_t const givendim, std::string const& basisOptionNames) : CLFException(), outdim(outdim), givendim(givendim), basisOptionNames(basisOptionNames) {
  if( outdim==1 & givendim==1 ) {
    message = "ERROR: SupportPoint requires " + std::to_string(outdim) + " basis, but " + std::to_string(givendim) + " was given. Given options: " + basisOptionNames;
  } else if( outdim==1 ) {
    message = "ERROR: SupportPoint requires " + std::to_string(outdim) + " basis, but " + std::to_string(givendim) + " were given. Given options: " + basisOptionNames;
  } else if( givendim==1 ) {
    message = "ERROR: SupportPoint requires " + std::to_string(outdim) + " bases, but " + std::to_string(givendim) + " was given. Given options: " + basisOptionNames;
  } else {
    message = "ERROR: SupportPoint requires " + std::to_string(outdim) + " bases, but " + std::to_string(givendim) + " were given. Given options: " + basisOptionNames;
  }
}

const std::vector<std::string> SupportPointInvalidBasisException::options =
    {
      "TOTALORDERPOLYNOMIALS",
      "TOTALORDERSINCOS"
    };

SupportPointInvalidBasisException::SupportPointInvalidBasisException(std::string const& basisType) : CLFException(), basisType(basisType) {
  message = "ERROR: SupportPoint tried to create the invalid basis type \"" + basisType + "\", options are:\n";
  for( const auto& it : options ) { message += it + "\n"; }
  message += "(clf::SupportPointInvalidBasisException).";
}
