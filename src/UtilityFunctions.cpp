#include "clf/UtilityFunctions.hpp"

using namespace clf;

std::string UtilityFunctions::ToUpper(std::string const& in) {
  std::string out = in;
  std::locale loc;
  for( std::string::size_type i=0; i<in.length(); ++i ) { out[i] = std::toupper(out[i], loc); }
  return out;
}
