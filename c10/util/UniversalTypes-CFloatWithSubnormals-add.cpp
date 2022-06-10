#include <c10/util/UniversalTypes-CFloatWithSubnormals-add.h>

namespace sw {
namespace universal {

// Suppress the warnings that __host__ functions were redeclared as __host__ __device__
#pragma diag_suppress 20040

// addition and necessary methods
template C10_HOST_DEVICE CFloat& CFloat::operator+=(const CFloat& rhs);
template C10_HOST_DEVICE void CFloat::normalizeAddition(
  BlockTripleAdd& tgt) const;
template C10_HOST_DEVICE int CFloat::scale() const noexcept;
template C10_HOST_DEVICE bool CFloat::isnormal() const noexcept;
template C10_HOST_DEVICE bool CFloat::isdenormal() const noexcept;
template C10_HOST_DEVICE uint64_t CFloat::fraction_ull() const;
template C10_HOST_DEVICE void CFloat::blockcopy<BlockTripleOperator::ADD>(
  BlockTripleAdd& tgt) const;
template<size_t srcbits, BlockTripleOperator op, size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating>
inline C10_HOST_DEVICE void convert(
  const blocktriple<srcbits, op, bt>& src,
  cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& tgt);
template C10_HOST_DEVICE CFloat& CFloat::operator++();

// blocktriple for ADD
template C10_HOST_DEVICE BlockTripleAdd::blocktriple() noexcept;
template C10_HOST_DEVICE bool BlockTripleAdd::any(size_t index) const noexcept;
template C10_HOST_DEVICE void BlockTripleAdd::clear() noexcept;
template C10_HOST_DEVICE bool BlockTripleAdd::sign() const noexcept;
template C10_HOST_DEVICE int BlockTripleAdd::scale() const noexcept;
template C10_HOST_DEVICE bool BlockTripleAdd::isnan() const noexcept;
template C10_HOST_DEVICE bool BlockTripleAdd::isinf() const noexcept;
template C10_HOST_DEVICE bool BlockTripleAdd::iszero() const noexcept;
template C10_HOST_DEVICE BlockTripleAdd::Significant
  BlockTripleAdd::significant() const noexcept;
template C10_HOST_DEVICE int BlockTripleAdd::significantscale() const noexcept;
template C10_HOST_DEVICE uint64_t BlockTripleAdd::significant_ull() const noexcept;
template C10_HOST_DEVICE void BlockTripleAdd::setnan(bool sign) noexcept;
template C10_HOST_DEVICE void BlockTripleAdd::setinf(bool sign) noexcept;
template C10_HOST_DEVICE void BlockTripleAdd::setzero(bool sign) noexcept;
template C10_HOST_DEVICE void BlockTripleAdd::setnormal() noexcept;
template C10_HOST_DEVICE void BlockTripleAdd::setsign(bool sign) noexcept;
template C10_HOST_DEVICE void BlockTripleAdd::setscale(int scale) noexcept;
template C10_HOST_DEVICE void BlockTripleAdd::setbits(uint64_t raw) noexcept;
template C10_HOST_DEVICE void BlockTripleAdd::setblock(size_t i, const uint32_t& block) noexcept;
template C10_HOST_DEVICE void BlockTripleAdd::setradix() noexcept;
#pragma push_macro("setbit")
#undef setbit
template C10_HOST_DEVICE void BlockTripleAdd::setbit(size_t index, bool v) noexcept;
#pragma pop_macro("setbit")
template C10_HOST_DEVICE BlockTripleAdd& BlockTripleAdd::bitShift(
  int leftShift) noexcept;
template C10_HOST_DEVICE std::pair<bool, size_t> BlockTripleAdd::roundingDecision(
  int adjustment) const noexcept;
template C10_HOST_DEVICE void BlockTripleAdd::add(
  BlockTripleAdd& lhs,
  BlockTripleAdd& rhs);

// blocksignificant for ADD
template C10_HOST_DEVICE BlockSignificantAdd::blocksignificant() noexcept;
template C10_HOST_DEVICE bool BlockSignificantAdd::any(size_t msb) const;
template C10_HOST_DEVICE void BlockSignificantAdd::clear() noexcept;
template C10_HOST_DEVICE bool BlockSignificantAdd::at(size_t bitIndex) const noexcept;
template C10_HOST_DEVICE bool BlockSignificantAdd::test(size_t bitIndex) const noexcept;
template C10_HOST_DEVICE uint64_t BlockSignificantAdd::significant_ull() const noexcept;
template C10_HOST_DEVICE bool BlockSignificantAdd::isneg() const;
template C10_HOST_DEVICE bool BlockSignificantAdd::sign() const;
template C10_HOST_DEVICE void BlockSignificantAdd::setradix(int radix);
template C10_HOST_DEVICE void BlockSignificantAdd::setbits(uint64_t value) noexcept;
template C10_HOST_DEVICE void BlockSignificantAdd::setblock(size_t b, const uint32_t& block) noexcept;
#pragma push_macro("setbit")
#undef setbit
template C10_HOST_DEVICE void BlockSignificantAdd::setbit(size_t index, bool v) noexcept;
#pragma pop_macro("setbit")
template C10_HOST_DEVICE BlockSignificantAdd& BlockSignificantAdd::operator<<=(int bitsToShift);
template C10_HOST_DEVICE bool BlockSignificantAdd::roundingDirection(size_t targetLsb) const;
template C10_HOST_DEVICE BlockSignificantAdd& BlockSignificantAdd::twosComplement() noexcept;
template C10_HOST_DEVICE BlockSignificantAdd& BlockSignificantAdd::flip() noexcept;
template C10_HOST_DEVICE void BlockSignificantAdd::add(
  const BlockSignificantAdd& lhs,
  const BlockSignificantAdd& rhs);
template C10_HOST_DEVICE int BlockSignificantAdd::msb() const noexcept;

#pragma diag_default 20040

}
}
