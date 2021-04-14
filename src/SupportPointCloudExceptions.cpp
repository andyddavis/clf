#include "clf/SupportPointCloudExceptions.hpp"

#include "clf/SupportPointCloud.hpp"

using namespace clf;

SupportPointCloudDimensionException::SupportPointCloudDimensionException(Type const& type, std::size_t const ind1, std::size_t const ind2) :
CLFException(),
type(type), ind1(ind1), ind2(ind2)
{
  if( type==Type::INPUT ) {
    message = "ERROR: SupportPointCloud was given two support points (SupportPoint type) with different input dimensions (clf::SupportPointCloudDimensionException).";
  } else if( type==Type::OUTPUT ) {
    message = "ERROR: SupportPointCloud was given two support points (SupportPoint type) with different output dimensions (clf::SupportPointCloudDimensionException).";
  }
}
