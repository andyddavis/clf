#include "clf/python/Pybind11Wrappers.hpp"

#include "clf/Parameters.hpp"

namespace py = pybind11;
using namespace clf;

template<typename TYPE>
void DefineAddGet(py::class_<Parameters, std::shared_ptr<Parameters> >& para) {
  para.def("Add", &Parameters::Add<TYPE>);
  para.def("Get", static_cast<TYPE (Parameters::*)(std::string const&)const>(&Parameters::Get<TYPE>));
}

void clf::python::ParametersWrapper(py::module& mod) {
  py::class_<Parameters, std::shared_ptr<Parameters> > para(mod, "Parameters");
  para.def(py::init<>());

  para.def("NumParameters", &Parameters::NumParameters);

  DefineAddGet<std::size_t>(para);
  DefineAddGet<double>(para);
}
