#include <cmath>
#include <type_traits>
#ifdef DIFFCONTACT_WITH_CPPAD
  #include <cppad/cppad.hpp>
#endif

namespace contactbench {
template <typename T>
typename std::enable_if<std::is_same<T, double>::value, double>::type
cos(const T &x) {
  return std::cos(x);
}

template <typename T>
typename std::enable_if<std::is_same<T, double>::value, double>::type
sin(const T &x) {
  return std::sin(x);
}

template <typename T>
typename std::enable_if<std::is_same<T, double>::value, double>::type
abs(const T &x) {
  return std::abs(x);
}

#ifdef DIFFCONTACT_WITH_CPPAD
template <typename T>
typename std::enable_if<std::is_same<T, CppAD::AD<double>>::value,
                        CppAD::AD<double>>::type
cos(const T &x) {
  return CppAD::cos(x);
}

template <typename T>
typename std::enable_if<std::is_same<T, CppAD::AD<double>>::value,
                        CppAD::AD<double>>::type
sin(const T &x) {
  return CppAD::sin(x);
}

template <typename T>
typename std::enable_if<std::is_same<T, CppAD::AD<double>>::value,
                        CppAD::AD<double>>::type
abs(const T &x) {
  return fabs(x);
}
#endif
} // namespace contactbench
