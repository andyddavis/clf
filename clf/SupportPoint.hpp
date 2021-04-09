#ifndef SUPPORTPOINT_HPP_
#define SUPPORTPOINT_HPP_

#include <boost/property_tree/ptree.hpp>

#include <MUQ/Modeling/ModPiece.h>

namespace clf {

/// The local function associated with a support point \f$x\f$.
/**
<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"OutputDimension"   | <tt>std::size_t</tt> | <tt>1</tt> | The output dimension of the support point. |
*/
class SupportPoint : public muq::Modeling::ModPiece {
public:

  /**
  @param[in] x The location of the support point \f$x\f$
  @param[in] pt The options for the support point
  */
  SupportPoint(Eigen::VectorXd const& x, boost::property_tree::ptree const& pt);

  virtual ~SupportPoint() = default;

  /// The input dimension \f$d\f$
  /**
  \return The input dimension \f$d\f$
  */
  std::size_t InputDimension() const;

  /// The output dimension \f$m\f$
  /**
  \return The output dimension \f$m\f$
  */
  std::size_t OutputDimension() const;

  /// The location of the support point \f$x\f$.
  const Eigen::VectorXd x;

private:

  /// Evaluate the local function \f$\ell\f$ associated with this support point
  /**
  Fills in the <tt>outputs</tt> vector attached to <tt>this</tt> SupportPoint (inherited from <tt>muq::Modeling::ModPiece</tt>). This is a vector of length <tt>1</tt> that stores the local function evaluation.
  @param[in] inputs There is only one input and it is the evaluation point
  */
  virtual void EvaluateImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& input) override;
};

} // namespace clf

#endif
