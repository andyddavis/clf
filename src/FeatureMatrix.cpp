#include "clf/FeatureMatrix.hpp"

using namespace clf;

FeatureMatrix::FeatureMatrix(std::shared_ptr<const FeatureVector> const& vec, std::size_t const numFeatureVectors) :
  numBasisFunctions(numFeatureVectors*vec->NumBasisFunctions()),
  numFeatureVectors(numFeatureVectors),
  featureVectors(std::vector<VectorPair>(1, VectorPair(vec, numFeatureVectors)))
{}

FeatureMatrix::FeatureMatrix(std::vector<VectorPair> const& featureVectors) :
  numBasisFunctions(ComputeNumBasisFunctions(featureVectors)),
  numFeatureVectors(ComputeNumFeatureVectors(featureVectors)),
  featureVectors(featureVectors)
{}

FeatureMatrix::FeatureMatrix(std::vector<std::shared_ptr<const FeatureVector> > const& vecs) :
  numBasisFunctions(ComputeNumBasisFunctions(vecs)),
  numFeatureVectors(vecs.size()),
  featureVectors(CreateVectorPairs(vecs))
{
}

std::vector<FeatureMatrix::VectorPair> FeatureMatrix::CreateVectorPairs(std::vector<std::shared_ptr<const FeatureVector> > const& vecs) {
  std::vector<VectorPair> pairs(vecs.size());
  for( std::size_t i=0; i<vecs.size(); ++i ) { pairs[i] = VectorPair(vecs[i], 1); }
  return pairs;
}

std::size_t FeatureMatrix::ComputeNumBasisFunctions(std::vector<std::shared_ptr<const FeatureVector> > const& vecs) {
  std::size_t num = 0;
  for( const auto& it : vecs ) { num += it->NumBasisFunctions(); }
  return num; 
}

std::size_t FeatureMatrix::ComputeNumBasisFunctions(std::vector<VectorPair> const& featureVectors) { 
  std::size_t num = 0;
  for( const auto& it : featureVectors ) { num += it.second*it.first->NumBasisFunctions(); }
  return num; 
}

std::size_t FeatureMatrix::ComputeNumFeatureVectors(std::vector<VectorPair> const& featureVectors) { 
  std::size_t num = 0;
  for( const auto& it : featureVectors ) { num += it.second; }
  return num; 
}

Eigen::VectorXd FeatureMatrix::ApplyTranspose(Eigen::VectorXd const& x, Eigen::VectorXd const& coeff) const {
  assert(coeff.size()==numBasisFunctions);
  Eigen::VectorXd output(numFeatureVectors);

  std::size_t count = 0;
  std::size_t start = 0;
  for( const auto& it : featureVectors ) {
    const Eigen::VectorXd phi = it.first->Evaluate(x);
    for( std::size_t i=0; i<it.second; ++i ) {
      output(count) = phi.dot(coeff.segment(start, phi.size()));

      ++count;
      start += it.first->NumBasisFunctions();
    }
  }
  assert(count==numFeatureVectors);
  assert(start==numBasisFunctions);

  return output;
}
