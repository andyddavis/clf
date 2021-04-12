#include "clf/CLFException.hpp"

using namespace clf;

CLFException::CLFException() : std::exception() {}

const char* CLFException::what() const noexcept { return message.c_str(); }
