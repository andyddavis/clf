#include "clf/SupportPointCloudExceptions.hpp"

using namespace clf;

SupportPointCloudDimensionExcpetion::SupportPointCloudDimensionExcpetion(Type const& type, std::size_t const ind1, std::size_t const ind2) :
std::exception(),
type(type), ind1(ind1), ind2(ind2)
{
  if( type==Type::INPUT ) {
    message = "ERROR: SupportPointCloud was given two support points (SupportPoint type) with different input dimensions.";
  } else if( type==Type::OUTPUT ) {
    message = "ERROR: SupportPointCloud was given two support points (SupportPoint type) with different output dimensions.";
  }
}

const char* SupportPointCloudDimensionExcpetion::what() const noexcept { return message.c_str(); }
