#include <c10/util/UniversalTypes.h>

namespace sw {
namespace universal {

// Suppress the warnings that __host__ functions were redeclared as __host__ __device__
#pragma diag_suppress 20040

// cfloat constructor
template C10_HOST_DEVICE CFloat::cfloat() noexcept;
template C10_HOST_DEVICE void CFloat::setblock(size_t b, uint32_t data) noexcept;

// blockbinary constructor and methods
template C10_HOST_DEVICE BlockBinary::blockbinary() noexcept;
template C10_HOST_DEVICE bool BlockBinary::isallones() const noexcept;
template C10_HOST_DEVICE bool BlockBinary::iszero() const noexcept;
template C10_HOST_DEVICE void BlockBinary::clear() noexcept;
template C10_HOST_DEVICE void BlockBinary::setbits(uint64_t value) noexcept;
#pragma push_macro("setbit")
#undef setbit
template C10_HOST_DEVICE void BlockBinary::setbit(size_t i, bool v) noexcept;
#pragma pop_macro("setbit")

// cfloat methods that handle blockbinary
template C10_HOST_DEVICE void CFloat::exponent(
  BlockBinary& e) const;

// isnan, isinf, and necessary methods
template C10_HOST_DEVICE bool CFloat::isnan(int NaNType) const noexcept;
template C10_HOST_DEVICE bool CFloat::isnanencoding(int NaNType) const noexcept;
template C10_HOST_DEVICE bool CFloat::issupernormal() const noexcept;
template C10_HOST_DEVICE bool CFloat::isinf(int InfType) const noexcept;
template C10_HOST_DEVICE bool CFloat::ispos() const noexcept;
template C10_HOST_DEVICE bool CFloat::isneg() const noexcept;
template C10_HOST_DEVICE bool CFloat::sign() const noexcept;

// Conversion to float and necessary methods
template C10_HOST_DEVICE CFloat::operator float() const noexcept;
template C10_HOST_DEVICE bool CFloat::iszero() const noexcept;
template C10_HOST_DEVICE bool CFloat::iszeroencoding() const noexcept;
template C10_HOST_DEVICE bool CFloat::at(size_t bitIndex) const noexcept;

// Conversion from float and necessary methods
template C10_HOST_DEVICE CFloat& CFloat::convert_ieee754<float>(float rhs) noexcept;
template C10_HOST_DEVICE void CFloat::setnan(int NaNType) noexcept;
template C10_HOST_DEVICE void CFloat::setinf(bool sign) noexcept;
template C10_HOST_DEVICE void CFloat::clear() noexcept;
template C10_HOST_DEVICE CFloat& CFloat::flip() noexcept;
template C10_HOST_DEVICE CFloat& CFloat::maxneg() noexcept;
template C10_HOST_DEVICE CFloat& CFloat::maxpos() noexcept;
template C10_HOST_DEVICE void CFloat::setsign(bool sign);
template C10_HOST_DEVICE void CFloat::shiftLeft(int leftShift);
template C10_HOST_DEVICE void CFloat::shiftRight(int rightShift);

// Explicit type casts to types other than float
template C10_HOST_DEVICE CFloat::operator int() const noexcept;
template C10_HOST_DEVICE CFloat::operator long() const noexcept;
template C10_HOST_DEVICE CFloat::operator long long() const noexcept;
template C10_HOST_DEVICE CFloat::operator double() const noexcept;
template C10_HOST_DEVICE int CFloat::to_int() const;
template C10_HOST_DEVICE long CFloat::to_long() const;
template C10_HOST_DEVICE long long CFloat::to_long_long() const;

// operator=
template C10_HOST_DEVICE CFloat& CFloat::operator=(float rhs) noexcept;

#pragma diag_default 20040

}
}
