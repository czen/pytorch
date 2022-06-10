#pragma once

// Do not include this file manually, include UniversalTypes.h instead

#include <c10/util/UniversalTypes-CFloatWithSubnormals-common.h>

namespace sw {
namespace universal {

#pragma diag_suppress 20040

// division and necessary methods
extern template C10_HOST_DEVICE CFloat& CFloat::operator/=(const CFloat& rhs);
extern template C10_HOST_DEVICE void CFloat::normalizeDivision(
  BlockTripleDiv& tgt) const;
extern template C10_HOST_DEVICE void CFloat::blockcopy<BlockTripleOperator::DIV>(
  BlockTripleDiv& tgt) const;

// blocktriple for DIV
extern template C10_HOST_DEVICE BlockTripleDiv::blocktriple() noexcept;
extern template C10_HOST_DEVICE bool BlockTripleDiv::any(size_t index) const noexcept;
extern template C10_HOST_DEVICE void BlockTripleDiv::clear() noexcept;
extern template C10_HOST_DEVICE bool BlockTripleDiv::sign() const noexcept;
extern template C10_HOST_DEVICE int BlockTripleDiv::scale() const noexcept;
extern template C10_HOST_DEVICE bool BlockTripleDiv::isnan() const noexcept;
extern template C10_HOST_DEVICE bool BlockTripleDiv::isinf() const noexcept;
extern template C10_HOST_DEVICE bool BlockTripleDiv::iszero() const noexcept;
extern template C10_HOST_DEVICE BlockTripleDiv::Significant
  BlockTripleDiv::significant() const noexcept;
extern template C10_HOST_DEVICE int BlockTripleDiv::significantscale() const noexcept;
extern template C10_HOST_DEVICE uint64_t BlockTripleDiv::significant_ull() const noexcept;
extern template C10_HOST_DEVICE void BlockTripleDiv::setnan(bool sign) noexcept;
extern template C10_HOST_DEVICE void BlockTripleDiv::setinf(bool sign) noexcept;
extern template C10_HOST_DEVICE void BlockTripleDiv::setzero(bool sign) noexcept;
extern template C10_HOST_DEVICE void BlockTripleDiv::setnormal() noexcept;
extern template C10_HOST_DEVICE void BlockTripleDiv::setsign(bool sign) noexcept;
extern template C10_HOST_DEVICE void BlockTripleDiv::setscale(int scale) noexcept;
extern template C10_HOST_DEVICE void BlockTripleDiv::setbits(uint64_t raw) noexcept;
extern template C10_HOST_DEVICE void BlockTripleDiv::setblock(size_t i, const uint32_t& block) noexcept;
extern template C10_HOST_DEVICE void BlockTripleDiv::setradix() noexcept;
#pragma push_macro("setbit")
#undef setbit
extern template C10_HOST_DEVICE void BlockTripleDiv::setbit(size_t index, bool v) noexcept;
#pragma pop_macro("setbit")
extern template C10_HOST_DEVICE BlockTripleDiv& BlockTripleDiv::bitShift(
  int leftShift) noexcept;
extern template C10_HOST_DEVICE std::pair<bool, size_t> BlockTripleDiv::roundingDecision(
  int adjustment) const noexcept;
extern template C10_HOST_DEVICE void BlockTripleDiv::div(
  BlockTripleDiv& lhs,
  BlockTripleDiv& rhs);

// blocksignificant for DIV
extern template C10_HOST_DEVICE BlockSignificantDiv::blocksignificant() noexcept;
extern template C10_HOST_DEVICE bool BlockSignificantDiv::any(size_t msb) const;
extern template C10_HOST_DEVICE void BlockSignificantDiv::clear() noexcept;
extern template C10_HOST_DEVICE bool BlockSignificantDiv::at(size_t bitIndex) const noexcept;
extern template C10_HOST_DEVICE bool BlockSignificantDiv::test(size_t bitIndex) const noexcept;
extern template C10_HOST_DEVICE uint64_t BlockSignificantDiv::significant_ull() const noexcept;
extern template C10_HOST_DEVICE bool BlockSignificantDiv::isneg() const;
extern template C10_HOST_DEVICE bool BlockSignificantDiv::sign() const;
extern template C10_HOST_DEVICE void BlockSignificantDiv::setradix(int radix);
extern template C10_HOST_DEVICE void BlockSignificantDiv::setbits(uint64_t value) noexcept;
extern template C10_HOST_DEVICE void BlockSignificantDiv::setblock(size_t b, const uint32_t& block) noexcept;
#pragma push_macro("setbit")
#undef setbit
extern template C10_HOST_DEVICE void BlockSignificantDiv::setbit(size_t index, bool v) noexcept;
#pragma pop_macro("setbit")
extern template C10_HOST_DEVICE BlockSignificantDiv& BlockSignificantDiv::operator<<=(int bitsToShift);
extern template C10_HOST_DEVICE bool BlockSignificantDiv::roundingDirection(size_t targetLsb) const;
extern template C10_HOST_DEVICE BlockSignificantDiv& BlockSignificantDiv::twosComplement() noexcept;
extern template C10_HOST_DEVICE BlockSignificantDiv& BlockSignificantDiv::flip() noexcept;
extern template C10_HOST_DEVICE void BlockSignificantDiv::div(
  const BlockSignificantDiv& lhs,
  const BlockSignificantDiv& rhs);
extern template C10_HOST_DEVICE int BlockSignificantDiv::msb() const noexcept;

#pragma diag_default 20040

}
}
