#pragma once

// Do not include this file manually, include UniversalTypes.h instead

#include <c10/util/UniversalTypes-CFloatWithSubnormals-common.h>

namespace sw {
namespace universal {

#pragma diag_suppress 20040

// addition and necessary methods
extern template C10_HOST_DEVICE CFloat& CFloat::operator+=(const CFloat& rhs);
extern template C10_HOST_DEVICE void CFloat::normalizeAddition(
  BlockTripleAdd& tgt) const;
extern template C10_HOST_DEVICE int CFloat::scale() const noexcept;
extern template C10_HOST_DEVICE bool CFloat::isnormal() const noexcept;
extern template C10_HOST_DEVICE bool CFloat::isdenormal() const noexcept;
extern template C10_HOST_DEVICE uint64_t CFloat::fraction_ull() const;
extern template C10_HOST_DEVICE void CFloat::blockcopy<BlockTripleOperator::ADD>(
  BlockTripleAdd& tgt) const;
extern template C10_HOST_DEVICE CFloat& CFloat::operator++();

// blocktriple for ADD
extern template C10_HOST_DEVICE BlockTripleAdd::blocktriple() noexcept;
extern template C10_HOST_DEVICE bool BlockTripleAdd::any(size_t index) const noexcept;
extern template C10_HOST_DEVICE void BlockTripleAdd::clear() noexcept;
extern template C10_HOST_DEVICE bool BlockTripleAdd::sign() const noexcept;
extern template C10_HOST_DEVICE int BlockTripleAdd::scale() const noexcept;
extern template C10_HOST_DEVICE bool BlockTripleAdd::isnan() const noexcept;
extern template C10_HOST_DEVICE bool BlockTripleAdd::isinf() const noexcept;
extern template C10_HOST_DEVICE bool BlockTripleAdd::iszero() const noexcept;
extern template C10_HOST_DEVICE BlockTripleAdd::Significant
  BlockTripleAdd::significant() const noexcept;
extern template C10_HOST_DEVICE int BlockTripleAdd::significantscale() const noexcept;
extern template C10_HOST_DEVICE uint64_t BlockTripleAdd::significant_ull() const noexcept;
extern template C10_HOST_DEVICE void BlockTripleAdd::setnan(bool sign) noexcept;
extern template C10_HOST_DEVICE void BlockTripleAdd::setinf(bool sign) noexcept;
extern template C10_HOST_DEVICE void BlockTripleAdd::setzero(bool sign) noexcept;
extern template C10_HOST_DEVICE void BlockTripleAdd::setnormal() noexcept;
extern template C10_HOST_DEVICE void BlockTripleAdd::setsign(bool sign) noexcept;
extern template C10_HOST_DEVICE void BlockTripleAdd::setscale(int scale) noexcept;
extern template C10_HOST_DEVICE void BlockTripleAdd::setbits(uint64_t raw) noexcept;
extern template C10_HOST_DEVICE void BlockTripleAdd::setblock(size_t i, const uint32_t& block) noexcept;
extern template C10_HOST_DEVICE void BlockTripleAdd::setradix() noexcept;
#pragma push_macro("setbit")
#undef setbit
extern template C10_HOST_DEVICE void BlockTripleAdd::setbit(size_t index, bool v) noexcept;
#pragma pop_macro("setbit")
extern template C10_HOST_DEVICE BlockTripleAdd& BlockTripleAdd::bitShift(
  int leftShift) noexcept;
extern template C10_HOST_DEVICE std::pair<bool, size_t> BlockTripleAdd::roundingDecision(
  int adjustment) const noexcept;
extern template C10_HOST_DEVICE void BlockTripleAdd::add(
  BlockTripleAdd& lhs,
  BlockTripleAdd& rhs);

// blocksignificant for ADD
extern template C10_HOST_DEVICE BlockSignificantAdd::blocksignificant() noexcept;
extern template C10_HOST_DEVICE bool BlockSignificantAdd::any(size_t msb) const;
extern template C10_HOST_DEVICE void BlockSignificantAdd::clear() noexcept;
extern template C10_HOST_DEVICE bool BlockSignificantAdd::at(size_t bitIndex) const noexcept;
extern template C10_HOST_DEVICE bool BlockSignificantAdd::test(size_t bitIndex) const noexcept;
extern template C10_HOST_DEVICE uint64_t BlockSignificantAdd::significant_ull() const noexcept;
extern template C10_HOST_DEVICE bool BlockSignificantAdd::isneg() const;
extern template C10_HOST_DEVICE bool BlockSignificantAdd::sign() const;
extern template C10_HOST_DEVICE void BlockSignificantAdd::setradix(int radix);
extern template C10_HOST_DEVICE void BlockSignificantAdd::setbits(uint64_t value) noexcept;
extern template C10_HOST_DEVICE void BlockSignificantAdd::setblock(size_t b, const uint32_t& block) noexcept;
#pragma push_macro("setbit")
#undef setbit
extern template C10_HOST_DEVICE void BlockSignificantAdd::setbit(size_t index, bool v) noexcept;
#pragma pop_macro("setbit")
extern template C10_HOST_DEVICE BlockSignificantAdd& BlockSignificantAdd::operator<<=(int bitsToShift);
extern template C10_HOST_DEVICE bool BlockSignificantAdd::roundingDirection(size_t targetLsb) const;
extern template C10_HOST_DEVICE BlockSignificantAdd& BlockSignificantAdd::twosComplement() noexcept;
extern template C10_HOST_DEVICE BlockSignificantAdd& BlockSignificantAdd::flip() noexcept;
extern template C10_HOST_DEVICE void BlockSignificantAdd::add(
  const BlockSignificantAdd& lhs,
  const BlockSignificantAdd& rhs);
extern template C10_HOST_DEVICE int BlockSignificantAdd::msb() const noexcept;

#pragma diag_default 20040

}
}
