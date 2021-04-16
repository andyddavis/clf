#ifndef SUPPORTPOINTEXCEPTIONS_HPP_
#define SUPPORTPOINTEXCEPTIONS_HPP_

#include <vector>

#include "clf/CLFException.hpp"

namespace clf {
namespace exceptions {

/// The support point does not use enough nearest neighbors to interpolate
class SupportPointWrongNumberOfNearestNeighbors : virtual public CLFException {
public:

  /**
  @param[in] output The output dimension that did not have enough nearest neighbors
  @param[in] required The number of neighbors required to interpolate
  @param[in] supplied The number of neighbors that the support point tried to use
  */
  SupportPointWrongNumberOfNearestNeighbors(std::size_t const output, std::size_t const required, std::size_t const supplied);

  virtual ~SupportPointWrongNumberOfNearestNeighbors() = default;

  /// The output dimension that did not have enough nearest neighbors
  const std::size_t output;

  /// The number of neighbors required to interpolate
  const std::size_t required;

  /// The number of neighbors that the support point tried to use
  const std::size_t supplied;
private:
};

/// The support point does not construct the right number of bases
class SupportPointWrongNumberOfBasesConstructed : virtual public CLFException {
public:

  /**
  @param[in] outdim The expected output dimension
  @param[in] givendim The given number of bases to create
  @param[in] basisOptionNames The names of the given bases
  */
  SupportPointWrongNumberOfBasesConstructed(std::size_t const outdim, std::size_t const givendim, std::string const& basisOptionNames);

  virtual ~SupportPointWrongNumberOfBasesConstructed() = default;

  /// The expected output dimension
  const std::size_t outdim;

  /// The given number of bases to create
  const std::size_t givendim;

  /// The names of the given bases
  const std::string basisOptionNames;

private:
};

/// The support point tried to create its basis functions in an invalid way
class SupportPointInvalidBasisException : virtual public CLFException {
public:

  /**
  @param[in] basisType The (invalid) basis that the support point tried to create
  */
  SupportPointInvalidBasisException(std::string const& basisType);

  virtual ~SupportPointInvalidBasisException() = default;

  /// The (invalid) basis that the support point tried to create
  const std::string basisType;

  /// The basis types that a support point can construct
  static const std::vector<std::string> options;

private:

};

} // namespace exceptions
} // namespace clf

#endif
