#pragma once

#include <universal/number/cfloat/cfloat.hpp>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace c10 {
    
#define FORALL_SUPPORTED_TYPES(_) \
  _(int)                          \
  _(long)                         \
  _(long long)                    \
  _(float)                        \
  _(double)                       \
  _(long double)
    

class CFloatWithSubnormals : public sw::universal::cfloat<32, 8, uint32_t, true, false, false>
{
public:
  using Base = sw::universal::cfloat<32, 8, uint32_t, true, false, false>;
  // Use all base constuctors
  using Base::Base;
  
  CFloatWithSubnormals(sw::universal::cfloat<32, 8, uint32_t, true, false, false> value)
  {
      *this = value;
  }
  
  // Implicit conversion operators
  #define OP(T) \
    inline operator T() const noexcept { return Base::operator T(); }
  FORALL_SUPPORTED_TYPES(OP)
  #undef OP
  inline operator bool() const noexcept { return *this != 0; }
  inline operator signed char() const noexcept { return to_native<signed char>(); }
  inline operator unsigned char() const noexcept { return to_native<unsigned char>(); }
  inline operator short() const noexcept { return to_native<short>(); }
  inline operator uint64_t() const noexcept { return to_native<uint64_t>(); }
  inline operator c10::BFloat16() const noexcept { return c10::BFloat16(*this); }
  inline operator c10::Half() const noexcept { return c10::Half(*this); }
  
  // Define operator < to avoid ambiguity
  #define OP(T) \
    inline bool operator<(T right)                  \
    {                                               \
      return sw::universal::operator<(              \
        *this,                                      \
        static_cast<CFloatWithSubnormals>(right));  \
    }                                               \
    inline bool operator<=(T right)                 \
    {                                               \
      return sw::universal::operator<=(             \
        *this,                                      \
        static_cast<CFloatWithSubnormals>(right));  \
    }
  FORALL_SUPPORTED_TYPES(OP)
  #undef OP
  #define OP(T) \
    inline friend bool operator<(T left, CFloatWithSubnormals right)  \
    {                                                                 \
      return left < static_cast<T>(right);                            \
    }                                                                 \
    inline friend bool operator<=(T left, CFloatWithSubnormals right) \
    {                                                                 \
      return left <= static_cast<T>(right);                           \
    }
  FORALL_SUPPORTED_TYPES(OP)
  #undef OP
  inline friend bool operator<(uint64_t left, CFloatWithSubnormals right)
  {
    return sw::universal::operator<(
      static_cast<CFloatWithSubnormals>(left),
      static_cast<CFloatWithSubnormals>(right));
  }
  inline friend bool operator<=(uint64_t left, CFloatWithSubnormals right)
  {
    return sw::universal::operator<=(
      static_cast<CFloatWithSubnormals>(left),
      static_cast<CFloatWithSubnormals>(right));
  }
  
  // Define operator > to avoid ambiguity
  #define OP(T) \
    inline bool operator>(T right)                  \
    {                                               \
      return sw::universal::operator>(              \
        *this,                                      \
        static_cast<CFloatWithSubnormals>(right));  \
    }                                               \
    inline bool operator>=(T right)                 \
    {                                               \
      return sw::universal::operator>=(             \
        *this,                                      \
        static_cast<CFloatWithSubnormals>(right));  \
    }
  FORALL_SUPPORTED_TYPES(OP)
  #undef OP
  #define OP(T) \
    inline friend bool operator>(T left, CFloatWithSubnormals right)  \
    {                                                                 \
      return left > static_cast<T>(right);                            \
    }                                                                 \
    inline friend bool operator>=(T left, CFloatWithSubnormals right) \
    {                                                                 \
      return left >= static_cast<T>(right);                           \
    }
  FORALL_SUPPORTED_TYPES(OP)
  #undef OP
  inline friend bool operator>(uint64_t left, CFloatWithSubnormals right)
  {
    return sw::universal::operator>(
      static_cast<CFloatWithSubnormals>(left),
      static_cast<CFloatWithSubnormals>(right));
  }
  inline friend bool operator>=(uint64_t left, CFloatWithSubnormals right)
  {
    return sw::universal::operator>=(
      static_cast<CFloatWithSubnormals>(left),
      static_cast<CFloatWithSubnormals>(right));
  }
  
  // Assignment operators
  #define OP(T) \
    inline CFloatWithSubnormals& operator=(T value){ Base::operator=(value); return *this; }
  FORALL_SUPPORTED_TYPES(OP)
  #undef OP
  inline CFloatWithSubnormals& operator=(signed char value)
  {
    convert_signed_integer(value);
    return *this;
  }
  inline CFloatWithSubnormals& operator=(unsigned char value)
  {
    convert_unsigned_integer(value);
    return *this;
  }
  inline CFloatWithSubnormals& operator=(short value)
  {
    convert_signed_integer(value);
    return *this;
  }
  inline CFloatWithSubnormals& operator=(bool value)
  {
    convert_signed_integer(value);
    return *this;
  }
  inline CFloatWithSubnormals& operator=(c10::BFloat16 value)
  {
    Base::operator=(static_cast<float>(value));
    return *this;
  }
  inline CFloatWithSubnormals& operator=(c10::Half value)
  {
    Base::operator=(static_cast<float>(value));
    return *this;
  }
  
  // Arithmetic operators
  inline CFloatWithSubnormals& operator+=(CFloatWithSubnormals right)
  {
    return static_cast<CFloatWithSubnormals&>(Base::operator+=(right));
  }
  inline CFloatWithSubnormals& operator-=(CFloatWithSubnormals right)
  {
    return static_cast<CFloatWithSubnormals&>(Base::operator-=(right));
  }
  inline CFloatWithSubnormals& operator*=(CFloatWithSubnormals right)
  {
    return static_cast<CFloatWithSubnormals&>(Base::operator*=(right));
  }
  inline CFloatWithSubnormals& operator/=(CFloatWithSubnormals right)
  {
    return static_cast<CFloatWithSubnormals&>(Base::operator/=(right));
  }
};

inline CFloatWithSubnormals operator+(CFloatWithSubnormals left, CFloatWithSubnormals right)
{
  return static_cast<CFloatWithSubnormals>(sw::universal::operator+(left, right));
}
inline CFloatWithSubnormals operator-(CFloatWithSubnormals left, CFloatWithSubnormals right)
{
  return static_cast<CFloatWithSubnormals>(sw::universal::operator-(left, right));
}
inline CFloatWithSubnormals operator*(CFloatWithSubnormals left, CFloatWithSubnormals right)
{
  return static_cast<CFloatWithSubnormals>(sw::universal::operator*(left, right));
}
inline CFloatWithSubnormals operator/(CFloatWithSubnormals left, CFloatWithSubnormals right)
{
  return static_cast<CFloatWithSubnormals>(sw::universal::operator/(left, right));
}

#define FORALL_SUPPORTED_TYPES_IN_OPERATORS(_) \
  _(int)                                       \
  _(float)                                     \
  _(double)

#define OP(T)                                                                        \
  inline CFloatWithSubnormals operator+(CFloatWithSubnormals left, T right)          \
  {                                                                                  \
    return static_cast<CFloatWithSubnormals>(sw::universal::operator+(left, right)); \
  }                                                                                  \
  inline CFloatWithSubnormals operator-(CFloatWithSubnormals left, T right)          \
  {                                                                                  \
    return static_cast<CFloatWithSubnormals>(sw::universal::operator-(left, right)); \
  }                                                                                  \
  inline CFloatWithSubnormals operator*(CFloatWithSubnormals left, T right)          \
  {                                                                                  \
    return static_cast<CFloatWithSubnormals>(sw::universal::operator*(left, right)); \
  }                                                                                  \
  inline CFloatWithSubnormals operator/(CFloatWithSubnormals left, T right)          \
  {                                                                                  \
    return static_cast<CFloatWithSubnormals>(sw::universal::operator/(left, right)); \
  }                                                                                  \
  inline CFloatWithSubnormals operator+(T left, CFloatWithSubnormals right)          \
  {                                                                                  \
    return static_cast<CFloatWithSubnormals>(sw::universal::operator+(left, right)); \
  }                                                                                  \
  inline CFloatWithSubnormals operator-(T left, CFloatWithSubnormals right)          \
  {                                                                                  \
    return static_cast<CFloatWithSubnormals>(sw::universal::operator-(left, right)); \
  }                                                                                  \
  inline CFloatWithSubnormals operator*(T left, CFloatWithSubnormals right)          \
  {                                                                                  \
    return static_cast<CFloatWithSubnormals>(sw::universal::operator*(left, right)); \
  }                                                                                  \
  inline CFloatWithSubnormals operator/(T left, CFloatWithSubnormals right)          \
  {                                                                                  \
    return static_cast<CFloatWithSubnormals>(sw::universal::operator/(left, right)); \
  }
FORALL_SUPPORTED_TYPES_IN_OPERATORS(OP)
#undef OP

#define OP(T)                                              \
  inline T& operator+=(T& left, CFloatWithSubnormals right) \
  {                                                        \
    return left += static_cast<T>(right);                  \
  }                                                        \
  inline T& operator-=(T& left, CFloatWithSubnormals right) \
  {                                                        \
    return left -= static_cast<T>(right);                  \
  }                                                        \
  inline T& operator*=(T& left, CFloatWithSubnormals right) \
  {                                                        \
    return left *= static_cast<T>(right);                  \
  }                                                        \
  inline T& operator/=(T& left, CFloatWithSubnormals right) \
  {                                                        \
    return left /= static_cast<T>(right);                  \
  }
FORALL_SUPPORTED_TYPES(OP)
#undef OP

inline CFloatWithSubnormals operator+(CFloatWithSubnormals left, int64_t right)
{
  CFloatWithSubnormals result(left);
  result += right;
  return result;
}
inline CFloatWithSubnormals operator-(CFloatWithSubnormals left, int64_t right)
{
  CFloatWithSubnormals result(left);
  result -= right;
  return result;
}
inline CFloatWithSubnormals operator*(CFloatWithSubnormals left, int64_t right)
{
  CFloatWithSubnormals result(left);
  result *= right;
  return result;
}
inline CFloatWithSubnormals operator/(CFloatWithSubnormals left, int64_t right)
{
  CFloatWithSubnormals result(left);
  result /= right;
  return result;
}
inline CFloatWithSubnormals operator+(int64_t left, CFloatWithSubnormals right)
{
  CFloatWithSubnormals result(left);
  result += right;
  return result;
}
inline CFloatWithSubnormals operator-(int64_t left, CFloatWithSubnormals right)
{
  CFloatWithSubnormals result(left);
  result -= right;
  return result;
}
inline CFloatWithSubnormals operator*(int64_t left, CFloatWithSubnormals right)
{
  CFloatWithSubnormals result(left);
  result *= right;
  return result;
}
inline CFloatWithSubnormals operator/(int64_t left, CFloatWithSubnormals right)
{
  CFloatWithSubnormals result(left);
  result /= right;
  return result;
}

#undef FORALL_SUPPORTED_TYPES
#undef FORALL_SUPPORTED_TYPES_IN_OPERATORS

}

namespace std {

template<>
struct hash<c10::CFloatWithSubnormals> {
  size_t operator()(const c10::CFloatWithSubnormals& value) const noexcept {
    return hash<uint32_t>()(reinterpret_cast<uint32_t>(value.block(0)));
  }
};

}
