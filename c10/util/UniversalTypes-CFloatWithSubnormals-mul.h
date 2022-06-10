#pragma once

// Do not include this file manually, include UniversalTypes.h instead

#include <c10/util/UniversalTypes-CFloatWithSubnormals-common.h>

namespace sw {
namespace universal {

#pragma diag_suppress 20040

// multiplication and necessary methods
extern template C10_HOST_DEVICE CFloat& CFloat::operator*=(const CFloat& rhs);
extern template C10_HOST_DEVICE void CFloat::normalizeMultiplication(
  BlockTripleMul& tgt) const;
extern template C10_HOST_DEVICE void CFloat::blockcopy<BlockTripleOperator::MUL>(
  BlockTripleMul& tgt) const;

// blocktriple for MUL
extern template C10_HOST_DEVICE BlockTripleMul::blocktriple() noexcept;
extern template C10_HOST_DEVICE bool BlockTripleMul::any(size_t index) const noexcept;
extern template C10_HOST_DEVICE void BlockTripleMul::clear() noexcept;
extern template C10_HOST_DEVICE bool BlockTripleMul::sign() const noexcept;
extern template C10_HOST_DEVICE int BlockTripleMul::scale() const noexcept;
extern template C10_HOST_DEVICE bool BlockTripleMul::isnan() const noexcept;
extern template C10_HOST_DEVICE bool BlockTripleMul::isinf() const noexcept;
extern template C10_HOST_DEVICE bool BlockTripleMul::iszero() const noexcept;
extern template C10_HOST_DEVICE BlockTripleMul::Significant
  BlockTripleMul::significant() const noexcept;
extern template C10_HOST_DEVICE int BlockTripleMul::significantscale() const noexcept;
extern template C10_HOST_DEVICE uint64_t BlockTripleMul::significant_ull() const noexcept;
extern template C10_HOST_DEVICE void BlockTripleMul::setnan(bool sign) noexcept;
extern template C10_HOST_DEVICE void BlockTripleMul::setinf(bool sign) noexcept;
extern template C10_HOST_DEVICE void BlockTripleMul::setzero(bool sign) noexcept;
extern template C10_HOST_DEVICE void BlockTripleMul::setnormal() noexcept;
extern template C10_HOST_DEVICE void BlockTripleMul::setsign(bool sign) noexcept;
extern template C10_HOST_DEVICE void BlockTripleMul::setscale(int scale) noexcept;
extern template C10_HOST_DEVICE void BlockTripleMul::setbits(uint64_t raw) noexcept;
extern template C10_HOST_DEVICE void BlockTripleMul::setblock(size_t i, const uint32_t& block) noexcept;
extern template C10_HOST_DEVICE void BlockTripleMul::setradix() noexcept;
#pragma push_macro("setbit")
#undef setbit
extern template C10_HOST_DEVICE void BlockTripleMul::setbit(size_t index, bool v) noexcept;
#pragma pop_macro("setbit")
extern template C10_HOST_DEVICE BlockTripleMul& BlockTripleMul::bitShift(
  int leftShift) noexcept;
extern template C10_HOST_DEVICE std::pair<bool, size_t> BlockTripleMul::roundingDecision(
  int adjustment) const noexcept;
extern template C10_HOST_DEVICE void BlockTripleMul::mul(
  BlockTripleMul& lhs,
  BlockTripleMul& rhs);

// blocksignificant for MUL
extern template C10_HOST_DEVICE BlockSignificantMul::blocksignificant() noexcept;
extern template C10_HOST_DEVICE bool BlockSignificantMul::any(size_t msb) const;
extern template C10_HOST_DEVICE void BlockSignificantMul::clear() noexcept;
extern template C10_HOST_DEVICE bool BlockSignificantMul::at(size_t bitIndex) const noexcept;
extern template C10_HOST_DEVICE bool BlockSignificantMul::test(size_t bitIndex) const noexcept;
extern template C10_HOST_DEVICE uint64_t BlockSignificantMul::significant_ull() const noexcept;
extern template C10_HOST_DEVICE bool BlockSignificantMul::isneg() const;
extern template C10_HOST_DEVICE bool BlockSignificantMul::sign() const;
extern template C10_HOST_DEVICE void BlockSignificantMul::setradix(int radix);
extern template C10_HOST_DEVICE void BlockSignificantMul::setbits(uint64_t value) noexcept;
extern template C10_HOST_DEVICE void BlockSignificantMul::setblock(size_t b, const uint32_t& block) noexcept;
#pragma push_macro("setbit")
#undef setbit
extern template C10_HOST_DEVICE void BlockSignificantMul::setbit(size_t index, bool v) noexcept;
#pragma pop_macro("setbit")
extern template C10_HOST_DEVICE BlockSignificantMul& BlockSignificantMul::operator<<=(int bitsToShift);
extern template C10_HOST_DEVICE bool BlockSignificantMul::roundingDirection(size_t targetLsb) const;
extern template C10_HOST_DEVICE BlockSignificantMul& BlockSignificantMul::twosComplement() noexcept;
extern template C10_HOST_DEVICE BlockSignificantMul& BlockSignificantMul::flip() noexcept;
extern template C10_HOST_DEVICE void BlockSignificantMul::mul(
  const BlockSignificantMul& lhs,
  const BlockSignificantMul& rhs);
extern template C10_HOST_DEVICE int BlockSignificantMul::msb() const noexcept;

#pragma diag_default 20040

}
}
