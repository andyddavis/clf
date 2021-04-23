#ifndef SUPPORTPOINT_HPP_
#define SUPPORTPOINT_HPP_

#include "clf/OptimizationOptions.hpp"
#include "clf/SupportPointBasis.hpp"
#include "clf/Model.hpp"
#include "clf/UncoupledCost.hpp"
#include "clf/SupportPointExceptions.hpp"

namespace clf {

/// Forward declaration of the clf::SupportPointCloud
class SupportPointCloud;

/// The local function \f$\ell\f$ associated with a support point \f$x\f$.
/**
Let \f$x \in \Omega \subseteq \mathbb{R}^{d}\f$ be a support point with an associated local function \f$\ell: \Omega \mapsto \mathbb{R}^{m}\f$. Suppose that we are building an approximation of the function \f$u:\Omega \mapsto \mathbb{R}^{m}\f$. Although \f$\ell\f$ is well-defined in the entire domain, we expect there is a smooth monotonic function \f$W: \Omega \mapsto \mathbb{R}^{+}\f$ with \f$W(0) \leq \epsilon\f$ and \f$W(r) \rightarrow \infty\f$ as \f$r \rightarrow \infty\f$ such that \f$\| \ell(y) - u(x) \|^2 \leq W(\|y-x\|^2)\f$. Therefore, we primarily care about the local function \f$\ell\f$ in a ball \f$\mathcal{B}_{\delta}(x)\f$ centered at \f$x\f$ with radius \f$\delta\f$.

Define the local coordinate \f$\hat{x}(y) = (y-x)/\delta\f$ (parameterized by \f$\delta\f$) and the basis functions for the \f$j^{th}\f$ output
\f{equation*}{
    \phi_j(y) = [\phi_1(\hat{x}(y)),\, \phi_2(\hat{x}(y)),\, ...,\, \phi_{q_j}(\hat{x}(y))]^{\top}.
\f}
The \f$j^{th}\f$ output of the local function is defined by coordinates \f$p_j \in \mathbb{R}^{q_j}\f$ such that \f$\ell(y) = \phi_j(y)^{\top} p\f$.

<B>Configuration Parameters:</B>
Parameter Key | Type | Default Value | Description |
------------- | ------------- | ------------- | ------------- |
"OutputDimension"   | <tt>std::size_t</tt> | <tt>1</tt> | The output dimension of the support point. |
"Radius"   | <tt>double</tt> | <tt>1.0</tt> | The initial value of the \f$\delta\f$ parameter. |
"RegularizationParameter"   | <tt>double</tt> | <tt>0.0</tt> | The regularization parameter for the uncoupled cost (see clf::UncoupledCost). |
"BasisFunctions"   | <tt>std::string</tt> |--- | The options to make the basis functions for each output, separated by commas (see SupportPoint::CreateBasisFunctions) |
"NumNeighbors"   | <tt>std::size_t</tt> | The number required to interpolate plus one | The number of nearest neighbors to use to compute the coefficients for each output. |
"Optimization"   | <tt>boost::property_tree::ptree</tt> | see clf::OptimizationOptions | The options for the uncoupled cost minimization |
*/
class SupportPoint {
// make the constructors private because we will always want to wrap the basis in a SupportPointBasis
private:
  /**
  @param[in] x The location of the support point \f$x\f$
  @param[in] model The model that defines the "data" at this support point
  @param[in] pt The options for the support point
  */
  SupportPoint(Eigen::VectorXd const& x, std::shared_ptr<const Model> const& model, boost::property_tree::ptree const& pt);

public:

  /**
  @param[in] x The location of the support point \f$x\f$
  @param[in] model The model that defines the "data" at this support point
  @param[in] pt The options for the support point
  \return A smart pointer to the support point
  */
  static std::shared_ptr<SupportPoint> Construct(Eigen::VectorXd const& x, std::shared_ptr<const Model> const& model, boost::property_tree::ptree const& pt);

  virtual ~SupportPoint() = default;

  /// Evaluate the nearest neighbor kernel at each neighboring support point
  /**
  \return The kernel evaluation at each support point
  */
  Eigen::VectorXd NearestNeighborKernel() const;

  /// Set the nearest neighbors
  /**
  @param[in] newcloud The clf::SupportPointCloud where this support point and its neighbors are stored
  @param[in] neighInd The global indices (in a clf::SupportPointCloud) of the nearest neighbors
  @param[in] neighDist The squared distances (Euclidean inner product) between the support point and its \f$j^{th}\f$ nearest neighbor
  */
  void SetNearestNeighbors(std::shared_ptr<const SupportPointCloud> const& newcloud, std::vector<std::size_t> const& neighInd, std::vector<double> const& neighDist);

  /// Get the global indices of this support points nearest neighbors
  /**
  \return The global indices of this support points nearest neighbors
  */
  std::vector<std::size_t> GlobalNeighborIndices() const;

  /// The number of coefficients associated with this support point
  /**
  The total number of coefficients is the sum of the coefficients associated with each basis.
  \return The number of coefficients associated with this support point
  */
  std::size_t NumCoefficients() const;

  /// Minimize the uncoupled cost function for this support point
  /**
  \return The uncoupled cost at the optimal coefficients value
  */
  double MinimizeUncoupledCost();

  /// The support point associated with the \f$j^{th}\f$ nearest neighbor
  /**
  @param[in] jnd The index of the \f$j^{th}\f$ nearest neighbor
  \return The point associated with \f$I(i,j)\f$
  */
  Eigen::VectorXd NearestNeighbor(std::size_t const jnd) const;

  /// Evaluate the operator applied to the local function at a given point
  /**
  @param[in] loc The location where we are evaluating the action of the operator
  @param[in] coefficients The coefficients that define the local function
  */
  Eigen::VectorXd Operator(Eigen::VectorXd const& loc, Eigen::VectorXd const& coefficients) const;

  /// Evaluate the Jacobian of the operator applied to the local function at a given point
  /**
  @param[in] loc The location where we are evaluating the action of the operator
  @param[in] coefficients The coefficients that define the local function
  */
  Eigen::MatrixXd OperatorJacobian(Eigen::VectorXd const& loc, Eigen::VectorXd const& coefficients) const;

  /// Evaluate the Hessian of the operator applied to the local function at a given point
  /**
  @param[in] loc The location where we are evaluating the action of the operator
  @param[in] coefficients The coefficients that define the local function
  */
  std::vector<Eigen::MatrixXd> OperatorHessian(Eigen::VectorXd const& loc, Eigen::VectorXd const& coefficients) const;

  /// Get the basis function
  /**
  \return Each component is the basis for the corresponding output
  */
  const std::vector<std::shared_ptr<const BasisFunctions> >& GetBasisFunctions() const;

  /// Get the number of nearest neighbors
  /**
  \return The number of nearest neighbors
  */
  std::size_t NumNeighbors() const;

  /// Evaluate the local function associated with this support point
  /**
  @param[in] loc The point where we want to evaluate the local function
  \return The function evaluation
  */
  Eigen::VectorXd EvaluateLocalFunction(Eigen::VectorXd const& loc) const;

  /// The global index of a neighbor given its local index
  /**
  @param[in] localInd The local neighbor index
  \return The global neighbor index
  */
  std::size_t GlobalNeighborIndex(std::size_t const localInd) const;

  /// The location of the support point \f$x\f$.
  const Eigen::VectorXd x;

  /// The model that defines the data/observations at this support point
  const std::shared_ptr<const Model> model;

private:

  /// Determine the number of nearest nieghbors for each output
  /**
  <B>Additional Configuration Parameters:</B>
  Parameter Key | Type | Default Value | Description |
  ------------- | ------------- | ------------- | ------------- |
  "LocalBasis"   | <tt>bool</tt> | <tt>true</tt> | Should we rescale the basis into local coordinates around the support point? |
  @param[in] bases The bases used for each output
  @param[in] pt The options for the support point
  \return The number of nearest neighbors used to compute the coefficients
  */
  static std::size_t DetermineNumNeighbors(std::vector<std::shared_ptr<const BasisFunctions> > const& bases, boost::property_tree::ptree const& pt);

  /// Create the basis functions from the given options
  /**
  @param[in] center The point that defines the center of the local function ball \f$\mathcal{B}_{\delta}(y)\f$ (the parameter \f$y\f$)
  @param[in] indim The input dimension for the support point
  @param[in] outdim The output dimension for the support point
  @param[in] pt The options for the basis functions
  \return The bases used for each output
  */
  static std::vector<std::shared_ptr<const BasisFunctions> > CreateBasisFunctions(std::shared_ptr<SupportPoint> const& point, std::size_t const indim, std::size_t const outdim, boost::property_tree::ptree pt);

  /// Create the basis functions from the given options
  /**
  @param[in] indim The input dimension for the support point
  @param[in] pt The options for the basis functions
  \return The basis created given the options
  */
  static std::shared_ptr<const BasisFunctions> CreateBasisFunctions(std::shared_ptr<SupportPoint> const& point, std::size_t const indim, boost::property_tree::ptree pt);

  /// Get the (optional) child ptree for the optimization options
  /**
  @param[in] pt The options given to this support point
  \return If specified, returns the options for minimizing the upcoupled cost. Otherwise, return an empty ptree and use the default options (see clf::OptimizationOptions)
  */
  static boost::property_tree::ptree GetOptimizationOptions(boost::property_tree::ptree const& pt);

  /// Minimize the uncoupled cost function for this support point using NLOPT
  /**
  \return The uncoupled cost at the optimal coefficients value
  */
  double MinimizeUncoupledCostNLOPT();

  /// Minimize the uncoupled cost function for this support point using Newton's method
  /**
  \return The uncoupled cost at the optimal coefficients value
  */
  double MinimizeUncoupledCostNewton();

  /// Optimization for the uncoupled cost minimization
  const OptimizationOptions optimizationOptions;

  /// The bases that defines this support point
  /**
  Each entry corresponds to one of the outputs. This vector has the same length as the number of outputs.

  Evaluating the \f$j^{th}\f$ entry defines the vector \f$\phi_j(y) = [\phi^{(0)}(\hat{x}(y)),\, \phi^{(1)}(\hat{x}(y)),\, ...,\, \phi^{(q_j)}(\hat{x}(y))]^{\top}\f$.
  */
  std::vector<std::shared_ptr<const BasisFunctions> > bases;

  /// The number of nearest neighbors used to compute the coefficients for each output
  std::size_t numNeighbors;

  /// The squared distances (Euclidean inner product) between the support point and its \f$j^{th}\f$ nearest neighbor
  std::vector<double> squaredNeighborDistances;

  /// The global indices (in a clf::SupportPointCloud) of the nearest neighbors
  std::vector<std::size_t> globalNeighorIndices;

  /// The cloud that stores this point and its nearest neighbor
  std::weak_ptr<const SupportPointCloud> cloud;

  /// The uncoupled cost function
  std::shared_ptr<UncoupledCost> uncoupledCost;

  /// The coefficients used to evaluate the local function associated with this support point
  Eigen::VectorXd coefficients;
};

} // namespace clf

#endif
