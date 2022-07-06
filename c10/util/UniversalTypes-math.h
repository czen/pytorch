#pragma once

#include <c10/util/math_compat.h>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")
#endif

// Math functions like in BFloat16-math.h
namespace std {

/// Used by vec256<c10::CFloatWithSubnormals>::map
inline c10::CFloatWithSubnormals acos(c10::CFloatWithSubnormals a) {
  return sw::universal::acos(a);
}
inline c10::CFloatWithSubnormals asin(c10::CFloatWithSubnormals a) {
  return sw::universal::asin(a);
}
inline c10::CFloatWithSubnormals atan(c10::CFloatWithSubnormals a) {
  return sw::universal::atan(a);
}
inline c10::CFloatWithSubnormals erf(c10::CFloatWithSubnormals a) {
  return sw::universal::erf(a);
}
inline c10::CFloatWithSubnormals erfc(c10::CFloatWithSubnormals a) {
  return sw::universal::erfc(a);
}
inline c10::CFloatWithSubnormals exp(c10::CFloatWithSubnormals a) {
  if (sw::universal::isinf(a)) {
    if (a < 0)
      return 0;
    else
      return a;
  }
  return sw::universal::exp(a);
}
inline c10::CFloatWithSubnormals expm1(c10::CFloatWithSubnormals a) {
  return sw::universal::expm1(a);
}
inline c10::CFloatWithSubnormals log(c10::CFloatWithSubnormals a) {
  return sw::universal::log(a);
}
inline c10::CFloatWithSubnormals log10(c10::CFloatWithSubnormals a) {
  return sw::universal::log10(a);
}
inline c10::CFloatWithSubnormals log1p(c10::CFloatWithSubnormals a) {
  return sw::universal::log1p(a);
}
inline c10::CFloatWithSubnormals log2(c10::CFloatWithSubnormals a) {
  return sw::universal::log2(a);
}
inline c10::CFloatWithSubnormals ceil(c10::CFloatWithSubnormals a) {
  return sw::universal::ceil(a);
}
inline c10::CFloatWithSubnormals cos(c10::CFloatWithSubnormals a) {
  return sw::universal::cos(a);
}
inline c10::CFloatWithSubnormals floor(c10::CFloatWithSubnormals a) {
  return sw::universal::floor(a);
}
inline c10::CFloatWithSubnormals nearbyint(c10::CFloatWithSubnormals a) {
  // FIXME
  return std::nearbyint(static_cast<float>(a));
}
inline c10::CFloatWithSubnormals sin(c10::CFloatWithSubnormals a) {
  return sw::universal::sin(a);
}
inline c10::CFloatWithSubnormals tan(c10::CFloatWithSubnormals a) {
  return sw::universal::tan(a);
}
inline c10::CFloatWithSubnormals sinh(c10::CFloatWithSubnormals a) {
  return sw::universal::sinh(a);
}
inline c10::CFloatWithSubnormals cosh(c10::CFloatWithSubnormals a) {
  return sw::universal::cosh(a);
}
inline c10::CFloatWithSubnormals tanh(c10::CFloatWithSubnormals a) {
  return sw::universal::tanh(a);
}
inline c10::CFloatWithSubnormals trunc(c10::CFloatWithSubnormals a) {
  return sw::universal::trunc(a);
}
inline c10::CFloatWithSubnormals lgamma(c10::CFloatWithSubnormals a) {
  // FIXME
  return std::lgamma(static_cast<float>(a));
}
inline c10::CFloatWithSubnormals sqrt(c10::CFloatWithSubnormals a) {
  return sw::universal::sqrt(a);
}
inline c10::CFloatWithSubnormals rsqrt(c10::CFloatWithSubnormals a) {
  return 1.0 / sw::universal::sqrt(a);
}
inline c10::CFloatWithSubnormals abs(c10::CFloatWithSubnormals a) {
  return sw::universal::abs(a);
}
inline c10::CFloatWithSubnormals pow(double a, c10::CFloatWithSubnormals b) {
  return sw::universal::pow(static_cast<c10::CFloatWithSubnormals>(a), b);
}
inline c10::CFloatWithSubnormals pow(c10::CFloatWithSubnormals a, double b) {
  return sw::universal::pow(a, b);
}
inline c10::CFloatWithSubnormals pow(c10::CFloatWithSubnormals a, c10::CFloatWithSubnormals b) {
  return sw::universal::pow(a, b);
}
inline c10::CFloatWithSubnormals fmod(c10::CFloatWithSubnormals a, c10::CFloatWithSubnormals b) {
  return sw::universal::fmod(a, b);
}
inline c10::CFloatWithSubnormals fabs(c10::CFloatWithSubnormals a) {
  return sw::universal::fabs(a);
}

C10_HOST_DEVICE inline c10::CFloatWithSubnormals nextafter(
    c10::CFloatWithSubnormals from,
    c10::CFloatWithSubnormals to) {
  return sw::universal::nextafter(from, to);
}

C10_HOST_DEVICE inline c10::CFloatWithSubnormals nexttoward(
    c10::CFloatWithSubnormals from,
    long double to) {
  // FIXME Universal does not have nexttoward yet
  return std::nexttoward(static_cast<float>(from), to);
}

} // namespace std

C10_CLANG_DIAGNOSTIC_POP()
