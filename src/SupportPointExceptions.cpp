#include "clf/SupportPointExceptions.hpp"

using namespace clf::exceptions;

SupportPointWrongNumberOfNearestNeighbors::SupportPointWrongNumberOfNearestNeighbors(std::size_t const output, std::size_t const required, std::size_t const supplied) : CLFException(), output(output), required(required), supplied(supplied) {
  if( required==1 & supplied==1 ) {
    message = "ERROR: SupportPoint output dimension " + std::to_string(output) + " requires " + std::to_string(required) + " point to interpolate but only " + std::to_string(supplied) + " was given (clf::exceptions::SupportPointWrongNumberOfNearestNeighbors).";
  } else if( required==1 ) {
    message = "ERROR: SupportPoint output dimension " + std::to_string(output) + " requires " + std::to_string(required) + " point to interpolate but only " + std::to_string(supplied) + " were given (clf::exceptions::SupportPointWrongNumberOfNearestNeighbors).";
  } else if ( supplied==1 ) {
    message = "ERROR: SupportPoint output dimension " + std::to_string(output) + " requires " + std::to_string(required) + " points to interpolate but only " + std::to_string(supplied) + " was given (clf::exceptions::SupportPointWrongNumberOfNearestNeighbors).";
  } else {
    message = "ERROR: SupportPoint output dimension " + std::to_string(output) + " requires " + std::to_string(required) + " points to interpolate but only " + std::to_string(supplied) + " were given (clf::exceptions::SupportPointWrongNumberOfNearestNeighbors).";
  }
}

SupportPointWrongNumberOfBasesConstructed::SupportPointWrongNumberOfBasesConstructed(std::size_t const outdim, std::size_t const givendim, std::string const& basisOptionNames) : CLFException(), outdim(outdim), givendim(givendim), basisOptionNames(basisOptionNames) {
  if( outdim==1 & givendim==1 ) {
    message = "ERROR: SupportPoint requires " + std::to_string(outdim) + " basis, but " + std::to_string(givendim) + " was given. Given options: " + basisOptionNames + " (clf::exceptions::SupportPointWrongNumberOfBasesConstructed).";
  } else if( outdim==1 ) {
    message = "ERROR: SupportPoint requires " + std::to_string(outdim) + " basis, but " + std::to_string(givendim) + " were given. Given options: " + basisOptionNames + " (clf::exceptions::SupportPointWrongNumberOfBasesConstructed).";
  } else if( givendim==1 ) {
    message = "ERROR: SupportPoint requires " + std::to_string(outdim) + " bases, but " + std::to_string(givendim) + " was given. Given options: " + basisOptionNames + " (clf::exceptions::SupportPointWrongNumberOfBasesConstructed).";
  } else {
    message = "ERROR: SupportPoint requires " + std::to_string(outdim) + " bases, but " + std::to_string(givendim) + " were given. Given options: " + basisOptionNames + " (clf::exceptions::SupportPointWrongNumberOfBasesConstructed).";
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
  message += "(clf::exceptions::SupportPointInvalidBasisException).";
}
