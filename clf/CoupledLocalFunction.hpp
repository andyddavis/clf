#ifndef COUPLEDLOCALFUNCTION_HPP_
#define COUPLEDLOCALFUNCTION_HPP_

#include "clf/SupportPointCloud.hpp"

namespace clf {

/// The coupled local function, which is an approximation \f$\hat{u} \approx u\f$ of a function \f$u: \Omega \mapsto \mathbb{R}^{m}\f$
/**
Let \f$\Omega \subseteq \mathbb{R}^{d}\f$ be the domain with closure \f$\overline{\Omega}\f$. Let \f$\mathcal{H}\f$ be a Hilbert space with inner product \f$\langle \cdot, \cdot \rangle_{\mathcal{H}}\f$ such that if \f$u \in \mathcal{H}\f$ then \f$u: \Omega \mapsto \mathbb{R}^{m}\f$.

A coupled local function is defined by a cloud of support points \f$\{x^{(i)}\}_{i=1}^{n}\f$ (see clf::SupportPointCloud) such that each support point is associated with a local function \f$\ell^{(i)}: \overline{\Omega} \mapsto \mathbb{R}^{m}\f$. The coupled local function is \f$\hat{u}(x) = \ell^{(I(x))}(x)\f$, where \f$I(x)\f$ is the index of the closest support point to \f$x\f$.
*/
class CoupledLocalFunction {
public:

  /**
  @param[in] cloud The support point cloud that stores all of the support points
  */
  CoupledLocalFunction(std::shared_ptr<SupportPointCloud> const& cloud);

  virtual ~CoupledLocalFunction() = default;
private:

  /// The support point cloud that stores all of the support points
  std::shared_ptr<SupportPointCloud> cloud;
};

} // namespace clf

#endif
