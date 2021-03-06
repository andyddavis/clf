#include <pybind11/pybind11.h>

#include <MUQ/Utilities/PyDictConversion.h>

#include "clf/Pybind11Wrappers.hpp"

#include "clf/Model.hpp"

namespace py = pybind11;
using namespace muq::Utilities;
using namespace clf;

namespace clf {
namespace python {
class PyModel : public Model {
public:
  //using ModelBase::ModelBase;
  inline PyModel(boost::property_tree::ptree const& pt) : Model(pt) {}
protected:
  inline virtual double RightHandSideComponentImpl(Eigen::VectorXd const& x, std::size_t const outind) const override { PYBIND11_OVERLOAD(double, Model, RightHandSideComponentImpl, x, outind); }

  inline virtual Eigen::VectorXd RightHandSideVectorImpl(Eigen::VectorXd const& x) const override { PYBIND11_OVERLOAD(Eigen::VectorXd, Model, RightHandSideVectorImpl, x); }

  inline virtual Eigen::VectorXd OperatorImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override { PYBIND11_OVERLOAD(Eigen::VectorXd, Model, OperatorImpl, x, coefficients, bases); }

  inline virtual Eigen::MatrixXd OperatorJacobianImpl(Eigen::VectorXd const& x, Eigen::VectorXd const& coefficients, std::vector<std::shared_ptr<const BasisFunctions> > const& bases) const override { PYBIND11_OVERLOAD(Eigen::MatrixXd, Model, OperatorJacobianImpl, x, coefficients, bases); }
public:
private:
};
} // namespace python
} // namespace clf

void clf::python::ModelWrapper(pybind11::module& mod) {
  py::class_<Model, PyModel, std::shared_ptr<Model> > model(mod, "Model");
  model.def(py::init( [] (py::dict const& d) { return new PyModel(ConvertDictToPtree(d)); }));
  model.def_readonly("inputDimension", &Model::inputDimension);
  model.def_readonly("outputDimension", &Model::outputDimension);
  model.def("RightHandSide", &Model::RightHandSide);
  model.def("OperatorJacobian", &Model::OperatorJacobian);
  model.def("OperatorJacobianByFD", &Model::OperatorJacobianByFD, "Compute the Jacobian using finite difference", py::arg("x"), py::arg("coefficients"), py::arg("bases"), py::arg("eval")=Eigen::VectorXd());
}
