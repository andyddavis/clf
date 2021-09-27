#ifndef LINEARMODEL_HPP_
#define LINEARMODEL_HPP_

#include "clf/Model.hpp"

namespace clf {

/// Implement a linear model with the form \f$\mathcal{L}(u) = L u\f$, where \f$L \in \mathbb{R}^{m \times n}\f$
template<typename MatrixType>
class LinearModel : public Model {
public:
  /// Construct a linear model, by default use the identity 
  inline LinearModel(boost::property_tree::ptree const& pt) : Model(pt) {}

  virtual ~LinearModel() = default;
private:
};
} // namespace clf

#endif
