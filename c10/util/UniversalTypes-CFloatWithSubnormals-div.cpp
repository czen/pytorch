#include <c10/util/UniversalTypes-CFloatWithSubnormals-div.h>

namespace sw {
namespace universal {

// Suppress the warnings that __host__ functions were redeclared as __host__ __device__
#pragma diag_suppress 20040

// division and necessary methods
template C10_HOST_DEVICE CFloat& CFloat::operator/=(const CFloat& rhs);
template C10_HOST_DEVICE void CFloat::normalizeDivision(
  BlockTripleDiv& tgt) const;
template C10_HOST_DEVICE void CFloat::blockcopy<BlockTripleOperator::DIV>(
  BlockTripleDiv& tgt) const;

// blocktriple for DIV
template C10_HOST_DEVICE BlockTripleDiv::blocktriple() noexcept;
template C10_HOST_DEVICE bool BlockTripleDiv::any(size_t index) const noexcept;
template C10_HOST_DEVICE void BlockTripleDiv::clear() noexcept;
template C10_HOST_DEVICE bool BlockTripleDiv::sign() const noexcept;
template C10_HOST_DEVICE int BlockTripleDiv::scale() const noexcept;
template C10_HOST_DEVICE bool BlockTripleDiv::isnan() const noexcept;
template C10_HOST_DEVICE bool BlockTripleDiv::isinf() const noexcept;
template C10_HOST_DEVICE bool BlockTripleDiv::iszero() const noexcept;
template C10_HOST_DEVICE BlockTripleDiv::Significant
  BlockTripleDiv::significant() const noexcept;
template C10_HOST_DEVICE int BlockTripleDiv::significantscale() const noexcept;
template C10_HOST_DEVICE uint64_t BlockTripleDiv::significant_ull() const noexcept;
template C10_HOST_DEVICE void BlockTripleDiv::setnan(bool sign) noexcept;
template C10_HOST_DEVICE void BlockTripleDiv::setinf(bool sign) noexcept;
template C10_HOST_DEVICE void BlockTripleDiv::setzero(bool sign) noexcept;
template C10_HOST_DEVICE void BlockTripleDiv::setnormal() noexcept;
template C10_HOST_DEVICE void BlockTripleDiv::setsign(bool sign) noexcept;
template C10_HOST_DEVICE void BlockTripleDiv::setscale(int scale) noexcept;
template C10_HOST_DEVICE void BlockTripleDiv::setbits(uint64_t raw) noexcept;
template C10_HOST_DEVICE void BlockTripleDiv::setblock(size_t i, const uint32_t& block) noexcept;
template C10_HOST_DEVICE void BlockTripleDiv::setradix() noexcept;
#pragma push_macro("setbit")
#undef setbit
template C10_HOST_DEVICE void BlockTripleDiv::setbit(size_t index, bool v) noexcept;
#pragma pop_macro("setbit")
template C10_HOST_DEVICE BlockTripleDiv& BlockTripleDiv::bitShift(
  int leftShift) noexcept;
template C10_HOST_DEVICE std::pair<bool, size_t> BlockTripleDiv::roundingDecision(
  int adjustment) const noexcept;
template C10_HOST_DEVICE void BlockTripleDiv::div(
  BlockTripleDiv& lhs,
  BlockTripleDiv& rhs);

// blocksignificant for DIV
template C10_HOST_DEVICE BlockSignificantDiv::blocksignificant() noexcept;
template C10_HOST_DEVICE bool BlockSignificantDiv::any(size_t msb) const;
template C10_HOST_DEVICE void BlockSignificantDiv::clear() noexcept;
template C10_HOST_DEVICE bool BlockSignificantDiv::at(size_t bitIndex) const noexcept;
template C10_HOST_DEVICE bool BlockSignificantDiv::test(size_t bitIndex) const noexcept;
template C10_HOST_DEVICE uint64_t BlockSignificantDiv::significant_ull() const noexcept;
template C10_HOST_DEVICE bool BlockSignificantDiv::isneg() const;
template C10_HOST_DEVICE bool BlockSignificantDiv::sign() const;
template C10_HOST_DEVICE void BlockSignificantDiv::setradix(int radix);
template C10_HOST_DEVICE void BlockSignificantDiv::setbits(uint64_t value) noexcept;
template C10_HOST_DEVICE void BlockSignificantDiv::setblock(size_t b, const uint32_t& block) noexcept;
#pragma push_macro("setbit")
#undef setbit
template C10_HOST_DEVICE void BlockSignificantDiv::setbit(size_t index, bool v) noexcept;
#pragma pop_macro("setbit")
template C10_HOST_DEVICE BlockSignificantDiv& BlockSignificantDiv::operator<<=(int bitsToShift);
template C10_HOST_DEVICE bool BlockSignificantDiv::roundingDirection(size_t targetLsb) const;
template C10_HOST_DEVICE BlockSignificantDiv& BlockSignificantDiv::twosComplement() noexcept;
template C10_HOST_DEVICE BlockSignificantDiv& BlockSignificantDiv::flip() noexcept;
template C10_HOST_DEVICE void BlockSignificantDiv::div(
  const BlockSignificantDiv& lhs,
  const BlockSignificantDiv& rhs);
template C10_HOST_DEVICE int BlockSignificantDiv::msb() const noexcept;

#pragma diag_default 20040

}
}
