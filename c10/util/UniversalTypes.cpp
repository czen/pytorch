#include <c10/util/UniversalTypes.h>

#include <limits>

namespace sw {
namespace universal {

// Suppress the warnings that __host__ functions were redeclared as __host__ __device__
#pragma diag_suppress 20040

// cfloat constructor
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>::cfloat() noexcept;
template C10_HOST_DEVICE void cfloat<32, 8, uint32_t, true, false, false>::setblock(size_t b, uint32_t data) noexcept;

// blockbinary constructor and methods
template C10_HOST_DEVICE blockbinary<8, uint32_t, BinaryNumberType::Signed>::blockbinary() noexcept;
template C10_HOST_DEVICE bool blockbinary<8, uint32_t, BinaryNumberType::Signed>::isallones() const noexcept;
template C10_HOST_DEVICE bool blockbinary<8, uint32_t, BinaryNumberType::Signed>::iszero() const noexcept;
template C10_HOST_DEVICE void blockbinary<8, uint32_t, BinaryNumberType::Signed>::clear() noexcept;
template C10_HOST_DEVICE void blockbinary<8, uint32_t, BinaryNumberType::Signed>::setbits(uint64_t value) noexcept;
#pragma push_macro("setbit")
#undef setbit
template C10_HOST_DEVICE void blockbinary<8, uint32_t, BinaryNumberType::Signed>::setbit(size_t i, bool v) noexcept;
#pragma pop_macro("setbit")

// cfloat methods that handle blockbinary
template C10_HOST_DEVICE void cfloat<32, 8, uint32_t, true, false, false>::exponent(
  blockbinary<8, uint32_t, BinaryNumberType::Signed>& e) const;

// isnan, isinf, and necessary methods
template C10_HOST_DEVICE bool cfloat<32, 8, uint32_t, true, false, false>::isnan(int NaNType) const noexcept;
template C10_HOST_DEVICE bool cfloat<32, 8, uint32_t, true, false, false>::isnanencoding(int NaNType) const noexcept;
template C10_HOST_DEVICE bool cfloat<32, 8, uint32_t, true, false, false>::issupernormal() const noexcept;
template C10_HOST_DEVICE bool cfloat<32, 8, uint32_t, true, false, false>::isinf(int InfType) const noexcept;
template C10_HOST_DEVICE bool cfloat<32, 8, uint32_t, true, false, false>::ispos() const noexcept;
template C10_HOST_DEVICE bool cfloat<32, 8, uint32_t, true, false, false>::isneg() const noexcept;
template C10_HOST_DEVICE bool cfloat<32, 8, uint32_t, true, false, false>::sign() const noexcept;

// Conversion to float and necessary methods
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>::operator float() const noexcept;
template C10_HOST_DEVICE bool cfloat<32, 8, uint32_t, true, false, false>::iszero() const noexcept;
template C10_HOST_DEVICE bool cfloat<32, 8, uint32_t, true, false, false>::iszeroencoding() const noexcept;
template C10_HOST_DEVICE bool cfloat<32, 8, uint32_t, true, false, false>::at(size_t bitIndex) const noexcept;

// Conversion from float and necessary methods
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>& cfloat<32, 8, uint32_t, true, false, false>::convert_ieee754<float>(float rhs) noexcept;
template C10_HOST_DEVICE void cfloat<32, 8, uint32_t, true, false, false>::setnan(int NaNType) noexcept;
template C10_HOST_DEVICE void cfloat<32, 8, uint32_t, true, false, false>::setinf(bool sign) noexcept;
template C10_HOST_DEVICE void cfloat<32, 8, uint32_t, true, false, false>::clear() noexcept;
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>& cfloat<32, 8, uint32_t, true, false, false>::flip() noexcept;
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>& cfloat<32, 8, uint32_t, true, false, false>::maxneg() noexcept;
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>& cfloat<32, 8, uint32_t, true, false, false>::maxpos() noexcept;
template C10_HOST_DEVICE void cfloat<32, 8, uint32_t, true, false, false>::setsign(bool sign);
template C10_HOST_DEVICE void cfloat<32, 8, uint32_t, true, false, false>::shiftLeft(int leftShift);
template C10_HOST_DEVICE void cfloat<32, 8, uint32_t, true, false, false>::shiftRight(int rightShift);

// Explicit type casts to types other than float
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>::operator int() const noexcept;
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>::operator long() const noexcept;
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>::operator long long() const noexcept;
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>::operator double() const noexcept;
template C10_HOST_DEVICE int cfloat<32, 8, uint32_t, true, false, false>::to_int() const;
template C10_HOST_DEVICE long cfloat<32, 8, uint32_t, true, false, false>::to_long() const;
template C10_HOST_DEVICE long long cfloat<32, 8, uint32_t, true, false, false>::to_long_long() const;

#pragma diag_default 20040

}
}
