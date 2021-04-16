#include "clf/SupportPointCloudExceptions.hpp"

#include "clf/SupportPointCloud.hpp"

using namespace clf::exceptions;

SupportPointCloudNotConnected::SupportPointCloudNotConnected(std::size_t const outnum) : CLFException(), outnum(outnum) {
  message = "ERROR: The graph in SupportPointCloud associated with output number " + std::to_string(outnum) + " is not connected (clf::exceptions::SupportPointCloudNotConnected).";
}

SupportPointCloudNotEnoughPointsException::SupportPointCloudNotEnoughPointsException(std::size_t const numPoints, std::size_t const required) : CLFException(), numPoints(numPoints), required(required) {
  if( numPoints==1 & required==1 ) {
    message = "ERROR: SupportPointCloud only has " + std::to_string(numPoints) + " support point but one of the support points requires " + std::to_string(required) + " nearest neighbor (clf::exceptions::SupportPointCloudNotEnoughPointsException).";
  } else if( numPoints==1 ) {
    message = "ERROR: SupportPointCloud only has " + std::to_string(numPoints) + " support point but one of the support points requires " + std::to_string(required) + " nearest neighbors (clf::exceptions::SupportPointCloudNotEnoughPointsException).";
  } else if( required==1 ) {
    message = "ERROR: SupportPointCloud only has " + std::to_string(numPoints) + " support points but one of the support points requires " + std::to_string(required) + " nearest neighbor (clf::exceptions::SupportPointCloudNotEnoughPointsException).";
  } else {
    message = "ERROR: SupportPointCloud only has " + std::to_string(numPoints) + " support points but one of the support points requires " + std::to_string(required) + " nearest neighbors (clf::exceptions::SupportPointCloudNotEnoughPointsException).";
  }
}

SupportPointCloudDimensionException::SupportPointCloudDimensionException(Type const& type, std::size_t const ind1, std::size_t const ind2) :
CLFException(),
type(type), ind1(ind1), ind2(ind2)
{
  if( type==Type::INPUT ) {
    message = "ERROR: SupportPointCloud was given two support points (SupportPoint type) with different input dimensions (clf::exceptions::SupportPointCloudDimensionException).";
  } else if( type==Type::OUTPUT ) {
    message = "ERROR: SupportPointCloud was given two support points (SupportPoint type) with different output dimensions (clf::exceptions::SupportPointCloudDimensionException).";
  }
}
