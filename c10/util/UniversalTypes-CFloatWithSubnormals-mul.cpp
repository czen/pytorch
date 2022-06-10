#include <c10/util/UniversalTypes-CFloatWithSubnormals-mul.h>

namespace sw {
namespace universal {

// Suppress the warnings that __host__ functions were redeclared as __host__ __device__
#pragma diag_suppress 20040

// multiplication and necessary methods
template C10_HOST_DEVICE CFloat& CFloat::operator*=(const CFloat& rhs);
template C10_HOST_DEVICE void CFloat::normalizeMultiplication(
  BlockTripleMul& tgt) const;
template C10_HOST_DEVICE void CFloat::blockcopy<BlockTripleOperator::MUL>(
  BlockTripleMul& tgt) const;

// blocktriple for MUL
template C10_HOST_DEVICE BlockTripleMul::blocktriple() noexcept;
template C10_HOST_DEVICE bool BlockTripleMul::any(size_t index) const noexcept;
template C10_HOST_DEVICE void BlockTripleMul::clear() noexcept;
template C10_HOST_DEVICE bool BlockTripleMul::sign() const noexcept;
template C10_HOST_DEVICE int BlockTripleMul::scale() const noexcept;
template C10_HOST_DEVICE bool BlockTripleMul::isnan() const noexcept;
template C10_HOST_DEVICE bool BlockTripleMul::isinf() const noexcept;
template C10_HOST_DEVICE bool BlockTripleMul::iszero() const noexcept;
template C10_HOST_DEVICE BlockTripleMul::Significant
  BlockTripleMul::significant() const noexcept;
template C10_HOST_DEVICE int BlockTripleMul::significantscale() const noexcept;
template C10_HOST_DEVICE uint64_t BlockTripleMul::significant_ull() const noexcept;
template C10_HOST_DEVICE void BlockTripleMul::setnan(bool sign) noexcept;
template C10_HOST_DEVICE void BlockTripleMul::setinf(bool sign) noexcept;
template C10_HOST_DEVICE void BlockTripleMul::setzero(bool sign) noexcept;
template C10_HOST_DEVICE void BlockTripleMul::setnormal() noexcept;
template C10_HOST_DEVICE void BlockTripleMul::setsign(bool sign) noexcept;
template C10_HOST_DEVICE void BlockTripleMul::setscale(int scale) noexcept;
template C10_HOST_DEVICE void BlockTripleMul::setbits(uint64_t raw) noexcept;
template C10_HOST_DEVICE void BlockTripleMul::setblock(size_t i, const uint32_t& block) noexcept;
template C10_HOST_DEVICE void BlockTripleMul::setradix() noexcept;
#pragma push_macro("setbit")
#undef setbit
template C10_HOST_DEVICE void BlockTripleMul::setbit(size_t index, bool v) noexcept;
#pragma pop_macro("setbit")
template C10_HOST_DEVICE BlockTripleMul& BlockTripleMul::bitShift(
  int leftShift) noexcept;
template C10_HOST_DEVICE std::pair<bool, size_t> BlockTripleMul::roundingDecision(
  int adjustment) const noexcept;
template C10_HOST_DEVICE void BlockTripleMul::mul(
  BlockTripleMul& lhs,
  BlockTripleMul& rhs);

// blocksignificant for MUL
template C10_HOST_DEVICE BlockSignificantMul::blocksignificant() noexcept;
template C10_HOST_DEVICE bool BlockSignificantMul::any(size_t msb) const;
template C10_HOST_DEVICE void BlockSignificantMul::clear() noexcept;
template C10_HOST_DEVICE bool BlockSignificantMul::at(size_t bitIndex) const noexcept;
template C10_HOST_DEVICE bool BlockSignificantMul::test(size_t bitIndex) const noexcept;
template C10_HOST_DEVICE uint64_t BlockSignificantMul::significant_ull() const noexcept;
template C10_HOST_DEVICE bool BlockSignificantMul::isneg() const;
template C10_HOST_DEVICE bool BlockSignificantMul::sign() const;
template C10_HOST_DEVICE void BlockSignificantMul::setradix(int radix);
template C10_HOST_DEVICE void BlockSignificantMul::setbits(uint64_t value) noexcept;
template C10_HOST_DEVICE void BlockSignificantMul::setblock(size_t b, const uint32_t& block) noexcept;
#pragma push_macro("setbit")
#undef setbit
template C10_HOST_DEVICE void BlockSignificantMul::setbit(size_t index, bool v) noexcept;
#pragma pop_macro("setbit")
template C10_HOST_DEVICE BlockSignificantMul& BlockSignificantMul::operator<<=(int bitsToShift);
template C10_HOST_DEVICE bool BlockSignificantMul::roundingDirection(size_t targetLsb) const;
template C10_HOST_DEVICE BlockSignificantMul& BlockSignificantMul::twosComplement() noexcept;
template C10_HOST_DEVICE BlockSignificantMul& BlockSignificantMul::flip() noexcept;
template C10_HOST_DEVICE void BlockSignificantMul::mul(
  const BlockSignificantMul& lhs,
  const BlockSignificantMul& rhs);
template C10_HOST_DEVICE int BlockSignificantMul::msb() const noexcept;

#pragma diag_default 20040

}
}
