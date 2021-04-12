#ifndef CLF_UTILITYFUNCTIONS_HPP_
#define CLF_UTILITYFUNCTIONS_HPP_

#include <string>
#include <locale>

namespace clf {

/// Utility functions for miscellaneous small but useful functions
class UtilityFunctions {
public:
  /// Convert a string to all upper case letters
  /**
  @param[in] in The input string
  \return The same string with all upper case letters
  */
  static std::string ToUpper(std::string const& in);
};

} // namespace clf

#endif
