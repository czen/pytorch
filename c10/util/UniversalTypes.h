#pragma once

#pragma push_macro("setbit")
#undef setbit
#include <universal/number/cfloat/cfloat.hpp>
#pragma pop_macro("setbit")

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
    
#define FORALL_SUPPORTED_TYPES(_) \
  _(int)                          \
  _(long)                         \
  _(long long)                    \
  _(float)                        \
  _(double)

#define FORALL_SUPPORTED_TYPES_IN_OPERATORS(_) \
  _(int)                                       \
  _(float)                                     \
  _(double)
  
#define FORALL_ADDITIONAL_TYPES(_) \
  _(unsigned char)                 \
  _(uint64_t)

// Redeclare operators as __host__ __device__
namespace sw {
namespace universal {

// Suppress the warnings that __host__ functions were redeclared as __host__ __device__
#pragma diag_suppress 20040

// Comparison
template<size_t nnbits, size_t nes, typename nbt, bool nsub, bool nsup, bool nsat>
inline C10_HOST_DEVICE bool operator<(const cfloat<nnbits, nes, nbt, nsub, nsup, nsat>& lhs, const cfloat<nnbits, nes, nbt, nsub, nsup, nsat>& rhs);

template<size_t nnbits, size_t nes, typename nbt, bool nsub, bool nsup, bool nsat>
inline C10_HOST_DEVICE bool operator<=(const cfloat<nnbits, nes, nbt, nsub, nsup, nsat>& lhs, const cfloat<nnbits, nes, nbt, nsub, nsup, nsat>& rhs);

template<size_t nnbits, size_t nes, typename nbt, bool nsub, bool nsup, bool nsat>
inline C10_HOST_DEVICE bool operator>(const cfloat<nnbits, nes, nbt, nsub, nsup, nsat>& lhs, const cfloat<nnbits, nes, nbt, nsub, nsup, nsat>& rhs);

template<size_t nnbits, size_t nes, typename nbt, bool nsub, bool nsup, bool nsat>
inline C10_HOST_DEVICE bool operator>=(const cfloat<nnbits, nes, nbt, nsub, nsup, nsat>& lhs, const cfloat<nnbits, nes, nbt, nsub, nsup, nsat>& rhs);

// Arithmetic
template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating>
inline C10_HOST_DEVICE cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> operator+(
  const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& lhs,
  const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& rhs);

template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating>
inline C10_HOST_DEVICE cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> operator-(
  const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& lhs,
  const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& rhs);

template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating>
inline C10_HOST_DEVICE cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> operator*(
  const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& lhs,
  const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& rhs);

template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating>
inline C10_HOST_DEVICE cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> operator/(
  const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& lhs,
  const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& rhs);


#define OP(T)                                                                                                 \
  template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating> \
  inline C10_HOST_DEVICE cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> operator+(       \
    T lhs,                                                                                                    \
    const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& rhs);                          \
                                                                                                              \
  template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating> \
  inline C10_HOST_DEVICE cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> operator-(       \
    T lhs,                                                                                                    \
    const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& rhs);                          \
                                                                                                              \
  template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating> \
  inline C10_HOST_DEVICE cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> operator*(       \
    T lhs,                                                                                                    \
    const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& rhs);                          \
                                                                                                              \
  template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating> \
  inline C10_HOST_DEVICE cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> operator/(       \
    T lhs,                                                                                                    \
    const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& rhs);                          \
                                                                                                              \
  template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating> \
  inline C10_HOST_DEVICE cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> operator+(       \
    const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& lhs,                           \
    T rhs);                                                                                                   \
                                                                                                              \
  template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating> \
  inline C10_HOST_DEVICE cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> operator-(       \
    const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& lhs,                           \
    T rhs);                                                                                                   \
                                                                                                              \
  template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating> \
  inline C10_HOST_DEVICE cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> operator*(       \
    const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& lhs,                           \
    T rhs);                                                                                                   \
                                                                                                              \
  template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating> \
  inline C10_HOST_DEVICE cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> operator/(       \
    const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& lhs,                           \
    T rhs);
FORALL_SUPPORTED_TYPES_IN_OPERATORS(OP)
#undef OP

// Instantiate cfloat methods and make them __host__ __device__
extern template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>::cfloat() noexcept;

extern template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>& cfloat<32, 8, uint32_t, true, false, false>::convert_ieee754<float>(float rhs) noexcept;

#pragma diag_default 20040

}
}

namespace c10 {
    

class CFloatWithSubnormals : public sw::universal::cfloat<32, 8, uint32_t, true, false, false>
{
public:
  using Base = sw::universal::cfloat<32, 8, uint32_t, true, false, false>;
  
  inline C10_HOST_DEVICE CFloatWithSubnormals() : Base() {}
  inline C10_HOST_DEVICE CFloatWithSubnormals(float value) : Base()
  {
    convert_ieee754<float>(value);
  }
  
  inline C10_HOST_DEVICE CFloatWithSubnormals(sw::universal::cfloat<32, 8, uint32_t, true, false, false> value) : Base(value) {}
  
  inline C10_HOST_DEVICE operator float() const noexcept { return Base::operator float(); }
  
  // Define operator < to avoid ambiguity
  #define OP(T)                                     \
    inline C10_HOST_DEVICE bool operator<(T right)  \
    {                                               \
      return sw::universal::operator<(              \
        static_cast<Base>(*this),                   \
        static_cast<Base>(right));                  \
    }                                               \
    inline C10_HOST_DEVICE bool operator<=(T right) \
    {                                               \
      return sw::universal::operator<=(             \
        static_cast<Base>(*this),                   \
        static_cast<Base>(right));                  \
    }
  FORALL_SUPPORTED_TYPES(OP)
  FORALL_ADDITIONAL_TYPES(OP)
  #undef OP
  #define OP(T) \
    inline C10_HOST_DEVICE friend bool operator<(T left, const CFloatWithSubnormals& right)  \
    {                                                                                        \
      return left < static_cast<T>(right);                                                   \
    }                                                                                        \
    inline C10_HOST_DEVICE friend bool operator<=(T left, const CFloatWithSubnormals& right) \
    {                                                                                        \
      return left <= static_cast<T>(right);                                                  \
    }
  FORALL_SUPPORTED_TYPES(OP)
  FORALL_ADDITIONAL_TYPES(OP)
  #undef OP
  
  // Define operator > to avoid ambiguity
  #define OP(T) \
    inline C10_HOST_DEVICE bool operator>(T right)  \
    {                                               \
      return sw::universal::operator>(              \
        static_cast<Base>(*this),                   \
        static_cast<Base>(right));                  \
    }                                               \
    inline C10_HOST_DEVICE bool operator>=(T right) \
    {                                               \
      return sw::universal::operator>=(             \
        static_cast<Base>(*this),                   \
        static_cast<Base>(right));                  \
    }
  FORALL_SUPPORTED_TYPES(OP)
  FORALL_ADDITIONAL_TYPES(OP)
  #undef OP
  #define OP(T) \
    inline C10_HOST_DEVICE friend bool operator>(T left, const CFloatWithSubnormals& right)  \
    {                                                                                        \
      return left > static_cast<T>(right);                                                   \
    }                                                                                        \
    inline C10_HOST_DEVICE friend bool operator>=(T left, const CFloatWithSubnormals& right) \
    {                                                                                        \
      return left >= static_cast<T>(right);                                                  \
    }
  FORALL_SUPPORTED_TYPES(OP)
  FORALL_ADDITIONAL_TYPES(OP)
  #undef OP
  
  // Assignment operators
  #define OP(T) \
    inline C10_HOST_DEVICE CFloatWithSubnormals& operator=(T value){ Base::operator=(value); return *this; }
  FORALL_SUPPORTED_TYPES(OP)
  #undef OP
  inline C10_HOST_DEVICE CFloatWithSubnormals& operator=(signed char value)
  {
    convert_signed_integer(value);
    return *this;
  }
  inline C10_HOST_DEVICE CFloatWithSubnormals& operator=(unsigned char value)
  {
    convert_unsigned_integer(value);
    return *this;
  }
  inline C10_HOST_DEVICE CFloatWithSubnormals& operator=(short value)
  {
    convert_signed_integer(value);
    return *this;
  }
  inline C10_HOST_DEVICE CFloatWithSubnormals& operator=(bool value)
  {
    convert_signed_integer(value);
    return *this;
  }
  inline C10_HOST_DEVICE CFloatWithSubnormals& operator=(c10::BFloat16 value)
  {
    Base::operator=(static_cast<float>(value));
    return *this;
  }
  inline C10_HOST_DEVICE CFloatWithSubnormals& operator=(c10::Half value)
  {
    Base::operator=(static_cast<float>(value));
    return *this;
  }
  
  // Arithmetic operators
  inline C10_HOST_DEVICE CFloatWithSubnormals& operator+=(const CFloatWithSubnormals& right)
  {
    return static_cast<CFloatWithSubnormals&>(Base::operator+=(right));
  }
  inline C10_HOST_DEVICE CFloatWithSubnormals& operator-=(const CFloatWithSubnormals& right)
  {
    return static_cast<CFloatWithSubnormals&>(Base::operator-=(right));
  }
  inline C10_HOST_DEVICE CFloatWithSubnormals& operator*=(const CFloatWithSubnormals& right)
  {
    return static_cast<CFloatWithSubnormals&>(Base::operator*=(right));
  }
  inline C10_HOST_DEVICE CFloatWithSubnormals& operator/=(const CFloatWithSubnormals& right)
  {
    return static_cast<CFloatWithSubnormals&>(Base::operator/=(right));
  }
};

inline C10_HOST_DEVICE CFloatWithSubnormals operator+(const CFloatWithSubnormals& left, const CFloatWithSubnormals& right)
{
  return static_cast<CFloatWithSubnormals>(sw::universal::operator+(
    static_cast<CFloatWithSubnormals::Base>(left),
    static_cast<CFloatWithSubnormals::Base>(right)));
}
inline C10_HOST_DEVICE CFloatWithSubnormals operator-(const CFloatWithSubnormals& left, const CFloatWithSubnormals& right)
{
  return static_cast<CFloatWithSubnormals>(sw::universal::operator-(
    static_cast<CFloatWithSubnormals::Base>(left),
    static_cast<CFloatWithSubnormals::Base>(right)));
}
inline C10_HOST_DEVICE CFloatWithSubnormals operator*(const CFloatWithSubnormals& left, const CFloatWithSubnormals& right)
{
  return static_cast<CFloatWithSubnormals>(sw::universal::operator*(
    static_cast<CFloatWithSubnormals::Base>(left),
    static_cast<CFloatWithSubnormals::Base>(right)));
}
inline C10_HOST_DEVICE CFloatWithSubnormals operator/(const CFloatWithSubnormals& left, const CFloatWithSubnormals& right)
{
  return static_cast<CFloatWithSubnormals>(sw::universal::operator/(
    static_cast<CFloatWithSubnormals::Base>(left),
    static_cast<CFloatWithSubnormals::Base>(right)));
}

#define OP(T)                                                                                                                 \
  inline C10_HOST_DEVICE CFloatWithSubnormals operator+(const CFloatWithSubnormals& left, T right)                            \
  {                                                                                                                           \
    return static_cast<CFloatWithSubnormals>(sw::universal::operator+(static_cast<CFloatWithSubnormals::Base>(left), right)); \
  }                                                                                                                           \
  inline C10_HOST_DEVICE CFloatWithSubnormals operator-(const CFloatWithSubnormals& left, T right)                            \
  {                                                                                                                           \
    return static_cast<CFloatWithSubnormals>(sw::universal::operator-(static_cast<CFloatWithSubnormals::Base>(left), right)); \
  }                                                                                                                           \
  inline C10_HOST_DEVICE CFloatWithSubnormals operator*(const CFloatWithSubnormals& left, T right)                            \
  {                                                                                                                           \
    return static_cast<CFloatWithSubnormals>(sw::universal::operator*(static_cast<CFloatWithSubnormals::Base>(left), right)); \
  }                                                                                                                           \
  inline C10_HOST_DEVICE CFloatWithSubnormals operator/(const CFloatWithSubnormals& left, T right)                            \
  {                                                                                                                           \
    return static_cast<CFloatWithSubnormals>(sw::universal::operator/(static_cast<CFloatWithSubnormals::Base>(left), right)); \
  }                                                                                                                           \
  inline C10_HOST_DEVICE CFloatWithSubnormals operator+(T left, const CFloatWithSubnormals& right)                            \
  {                                                                                                                           \
    return static_cast<CFloatWithSubnormals>(sw::universal::operator+(left, static_cast<CFloatWithSubnormals::Base>(right))); \
  }                                                                                                                           \
  inline C10_HOST_DEVICE CFloatWithSubnormals operator-(T left, const CFloatWithSubnormals& right)                            \
  {                                                                                                                           \
    return static_cast<CFloatWithSubnormals>(sw::universal::operator-(left, static_cast<CFloatWithSubnormals::Base>(right))); \
  }                                                                                                                           \
  inline C10_HOST_DEVICE CFloatWithSubnormals operator*(T left, const CFloatWithSubnormals& right)                            \
  {                                                                                                                           \
    return static_cast<CFloatWithSubnormals>(sw::universal::operator*(left, static_cast<CFloatWithSubnormals::Base>(right))); \
  }                                                                                                                           \
  inline C10_HOST_DEVICE CFloatWithSubnormals operator/(T left, const CFloatWithSubnormals& right)                            \
  {                                                                                                                           \
    return static_cast<CFloatWithSubnormals>(sw::universal::operator/(left, static_cast<CFloatWithSubnormals::Base>(right))); \
  }
FORALL_SUPPORTED_TYPES_IN_OPERATORS(OP)
#undef OP

#define OP(T)                                                                      \
  inline C10_HOST_DEVICE T& operator+=(T& left, const CFloatWithSubnormals& right) \
  {                                                                                \
    return left += static_cast<T>(right);                                          \
  }                                                                                \
  inline C10_HOST_DEVICE T& operator-=(T& left, const CFloatWithSubnormals& right) \
  {                                                                                \
    return left -= static_cast<T>(right);                                          \
  }                                                                                \
  inline C10_HOST_DEVICE T& operator*=(T& left, const CFloatWithSubnormals& right) \
  {                                                                                \
    return left *= static_cast<T>(right);                                          \
  }                                                                                \
  inline C10_HOST_DEVICE T& operator/=(T& left, const CFloatWithSubnormals& right) \
  {                                                                                \
    return left /= static_cast<T>(right);                                          \
  }
FORALL_SUPPORTED_TYPES(OP)
#undef OP

inline C10_HOST_DEVICE CFloatWithSubnormals operator+(const CFloatWithSubnormals& left, int64_t right)
{
  CFloatWithSubnormals result(left);
  result += right;
  return result;
}
inline C10_HOST_DEVICE CFloatWithSubnormals operator-(const CFloatWithSubnormals& left, int64_t right)
{
  CFloatWithSubnormals result(left);
  result -= right;
  return result;
}
inline C10_HOST_DEVICE CFloatWithSubnormals operator*(const CFloatWithSubnormals& left, int64_t right)
{
  CFloatWithSubnormals result(left);
  result *= right;
  return result;
}
inline C10_HOST_DEVICE CFloatWithSubnormals operator/(const CFloatWithSubnormals& left, int64_t right)
{
  CFloatWithSubnormals result(left);
  result /= right;
  return result;
}
inline C10_HOST_DEVICE CFloatWithSubnormals operator+(int64_t left, const CFloatWithSubnormals& right)
{
  CFloatWithSubnormals result(left);
  result += right;
  return result;
}
inline C10_HOST_DEVICE CFloatWithSubnormals operator-(int64_t left, const CFloatWithSubnormals& right)
{
  CFloatWithSubnormals result(left);
  result -= right;
  return result;
}
inline C10_HOST_DEVICE CFloatWithSubnormals operator*(int64_t left, const CFloatWithSubnormals& right)
{
  CFloatWithSubnormals result(left);
  result *= right;
  return result;
}
inline C10_HOST_DEVICE CFloatWithSubnormals operator/(int64_t left, const CFloatWithSubnormals& right)
{
  CFloatWithSubnormals result(left);
  result /= right;
  return result;
}

#undef FORALL_SUPPORTED_TYPES
#undef FORALL_SUPPORTED_TYPES_IN_OPERATORS
#undef FORALL_ADDITIONAL_TYPES

}

namespace std {

template<>
struct hash<c10::CFloatWithSubnormals> {
  size_t operator()(const c10::CFloatWithSubnormals& value) const noexcept {
    return hash<uint32_t>()(reinterpret_cast<uint32_t>(value.block(0)));
  }
};

}
