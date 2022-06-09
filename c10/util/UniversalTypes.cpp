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

// Conversion from double
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>&
  cfloat<32, 8, uint32_t, true, false, false>::convert_ieee754<double>(double rhs) noexcept;

// Explicit type casts to types other than float
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>::operator int() const noexcept;
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>::operator long() const noexcept;
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>::operator long long() const noexcept;
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>::operator double() const noexcept;
template C10_HOST_DEVICE int cfloat<32, 8, uint32_t, true, false, false>::to_int() const;
template C10_HOST_DEVICE long cfloat<32, 8, uint32_t, true, false, false>::to_long() const;
template C10_HOST_DEVICE long long cfloat<32, 8, uint32_t, true, false, false>::to_long_long() const;

// operator=
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>& cfloat<32, 8, uint32_t, true, false, false>::operator=(
  float rhs) noexcept;
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>& cfloat<32, 8, uint32_t, true, false, false>::operator=(
  double rhs) noexcept;

// addition and necessary methods
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>& cfloat<32, 8, uint32_t, true, false, false>::operator+=(
  const cfloat<32, 8, uint32_t, true, false, false>& rhs);
template C10_HOST_DEVICE void cfloat<32, 8, uint32_t, true, false, false>::normalizeAddition(
  blocktriple<23, BlockTripleOperator::ADD, uint32_t>& tgt) const;
template C10_HOST_DEVICE int cfloat<32, 8, uint32_t, true, false, false>::scale() const noexcept;
template C10_HOST_DEVICE bool cfloat<32, 8, uint32_t, true, false, false>::isnormal() const noexcept;
template C10_HOST_DEVICE bool cfloat<32, 8, uint32_t, true, false, false>::isdenormal() const noexcept;
template C10_HOST_DEVICE uint64_t cfloat<32, 8, uint32_t, true, false, false>::fraction_ull() const;
template C10_HOST_DEVICE void cfloat<32, 8, uint32_t, true, false, false>::blockcopy<BlockTripleOperator::ADD>(
  blocktriple<23, BlockTripleOperator::ADD, uint32_t>& tgt) const;
template<size_t srcbits, BlockTripleOperator op, size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating>
inline C10_HOST_DEVICE void convert(
  const blocktriple<srcbits, op, bt>& src,
  cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& tgt);

// blocktriple for ADD
template C10_HOST_DEVICE blocktriple<23, BlockTripleOperator::ADD, uint32_t>::blocktriple() noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::ADD, uint32_t>::clear() noexcept;
template C10_HOST_DEVICE bool blocktriple<23, BlockTripleOperator::ADD, uint32_t>::sign() const noexcept;
template C10_HOST_DEVICE int blocktriple<23, BlockTripleOperator::ADD, uint32_t>::scale() const noexcept;
template C10_HOST_DEVICE bool blocktriple<23, BlockTripleOperator::ADD, uint32_t>::isnan() const noexcept;
template C10_HOST_DEVICE bool blocktriple<23, BlockTripleOperator::ADD, uint32_t>::isinf() const noexcept;
template C10_HOST_DEVICE bool blocktriple<23, BlockTripleOperator::ADD, uint32_t>::iszero() const noexcept;
template C10_HOST_DEVICE blocktriple<23, BlockTripleOperator::ADD, uint32_t>::Significant
  blocktriple<23, BlockTripleOperator::ADD, uint32_t>::significant() const noexcept;
template C10_HOST_DEVICE int blocktriple<23, BlockTripleOperator::ADD, uint32_t>::significantscale() const noexcept;
template C10_HOST_DEVICE uint64_t blocktriple<23, BlockTripleOperator::ADD, uint32_t>::significant_ull() const noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::ADD, uint32_t>::setnan(bool sign) noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::ADD, uint32_t>::setinf(bool sign) noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::ADD, uint32_t>::setzero(bool sign) noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::ADD, uint32_t>::setnormal() noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::ADD, uint32_t>::setsign(bool sign) noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::ADD, uint32_t>::setscale(int scale) noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::ADD, uint32_t>::setbits(uint64_t raw) noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::ADD, uint32_t>::setblock(size_t i, const uint32_t& block) noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::ADD, uint32_t>::setradix() noexcept;
#pragma push_macro("setbit")
#undef setbit
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::ADD, uint32_t>::setbit(size_t index, bool v) noexcept;
#pragma pop_macro("setbit")
template C10_HOST_DEVICE blocktriple<23, BlockTripleOperator::ADD, uint32_t>& blocktriple<23, BlockTripleOperator::ADD, uint32_t>::bitShift(
  int leftShift) noexcept;
template C10_HOST_DEVICE std::pair<bool, size_t> blocktriple<23, BlockTripleOperator::ADD, uint32_t>::roundingDecision(
  int adjustment) const noexcept;

// blocksignificant
template C10_HOST_DEVICE blocksignificant<29, uint32_t>::blocksignificant() noexcept;
template C10_HOST_DEVICE void blocksignificant<29, uint32_t>::clear() noexcept;
template C10_HOST_DEVICE bool blocksignificant<29, uint32_t>::at(size_t bitIndex) const noexcept;
template C10_HOST_DEVICE uint64_t blocksignificant<29, uint32_t>::significant_ull() const noexcept;
template C10_HOST_DEVICE void blocksignificant<29, uint32_t>::setradix(int radix);
template C10_HOST_DEVICE void blocksignificant<29, uint32_t>::setbits(uint64_t value) noexcept;
template C10_HOST_DEVICE void blocksignificant<29, uint32_t>::setblock(size_t b, const uint32_t& block) noexcept;
#pragma push_macro("setbit")
#undef setbit
template C10_HOST_DEVICE void blocksignificant<29, uint32_t>::setbit(size_t index, bool v) noexcept;
#pragma pop_macro("setbit")
template C10_HOST_DEVICE blocksignificant<29, uint32_t>& blocksignificant<29, uint32_t>::operator<<=(int bitsToShift);
template C10_HOST_DEVICE bool blocksignificant<29, uint32_t>::roundingDirection(size_t targetLsb) const;

// multiplication and necessary methods
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>& cfloat<32, 8, uint32_t, true, false, false>::operator*=(
  const cfloat<32, 8, uint32_t, true, false, false>& rhs);
template C10_HOST_DEVICE void cfloat<32, 8, uint32_t, true, false, false>::normalizeMultiplication(
  blocktriple<23, BlockTripleOperator::MUL, uint32_t>& tgt) const;
template C10_HOST_DEVICE void cfloat<32, 8, uint32_t, true, false, false>::blockcopy<BlockTripleOperator::MUL>(
  blocktriple<23, BlockTripleOperator::MUL, uint32_t>& tgt) const;

// blocktriple for MUL
template C10_HOST_DEVICE blocktriple<23, BlockTripleOperator::MUL, uint32_t>::blocktriple() noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::MUL, uint32_t>::clear() noexcept;
template C10_HOST_DEVICE bool blocktriple<23, BlockTripleOperator::MUL, uint32_t>::sign() const noexcept;
template C10_HOST_DEVICE int blocktriple<23, BlockTripleOperator::MUL, uint32_t>::scale() const noexcept;
template C10_HOST_DEVICE bool blocktriple<23, BlockTripleOperator::MUL, uint32_t>::isnan() const noexcept;
template C10_HOST_DEVICE bool blocktriple<23, BlockTripleOperator::MUL, uint32_t>::isinf() const noexcept;
template C10_HOST_DEVICE bool blocktriple<23, BlockTripleOperator::MUL, uint32_t>::iszero() const noexcept;
template C10_HOST_DEVICE blocktriple<23, BlockTripleOperator::MUL, uint32_t>::Significant
  blocktriple<23, BlockTripleOperator::MUL, uint32_t>::significant() const noexcept;
template C10_HOST_DEVICE int blocktriple<23, BlockTripleOperator::MUL, uint32_t>::significantscale() const noexcept;
template C10_HOST_DEVICE uint64_t blocktriple<23, BlockTripleOperator::MUL, uint32_t>::significant_ull() const noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::MUL, uint32_t>::setnan(bool sign) noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::MUL, uint32_t>::setinf(bool sign) noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::MUL, uint32_t>::setzero(bool sign) noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::MUL, uint32_t>::setnormal() noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::MUL, uint32_t>::setsign(bool sign) noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::MUL, uint32_t>::setscale(int scale) noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::MUL, uint32_t>::setbits(uint64_t raw) noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::MUL, uint32_t>::setblock(size_t i, const uint32_t& block) noexcept;
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::MUL, uint32_t>::setradix() noexcept;
#pragma push_macro("setbit")
#undef setbit
template C10_HOST_DEVICE void blocktriple<23, BlockTripleOperator::MUL, uint32_t>::setbit(size_t index, bool v) noexcept;
#pragma pop_macro("setbit")
template C10_HOST_DEVICE blocktriple<23, BlockTripleOperator::MUL, uint32_t>& blocktriple<23, BlockTripleOperator::MUL, uint32_t>::bitShift(
  int leftShift) noexcept;
template C10_HOST_DEVICE std::pair<bool, size_t> blocktriple<23, BlockTripleOperator::MUL, uint32_t>::roundingDecision(
  int adjustment) const noexcept;

// blocksignificant
template C10_HOST_DEVICE blocksignificant<48, uint32_t>::blocksignificant() noexcept;
template C10_HOST_DEVICE void blocksignificant<48, uint32_t>::clear() noexcept;
template C10_HOST_DEVICE bool blocksignificant<48, uint32_t>::at(size_t bitIndex) const noexcept;
template C10_HOST_DEVICE uint64_t blocksignificant<48, uint32_t>::significant_ull() const noexcept;
template C10_HOST_DEVICE void blocksignificant<48, uint32_t>::setradix(int radix);
template C10_HOST_DEVICE void blocksignificant<48, uint32_t>::setbits(uint64_t value) noexcept;
template C10_HOST_DEVICE void blocksignificant<48, uint32_t>::setblock(size_t b, const uint32_t& block) noexcept;
#pragma push_macro("setbit")
#undef setbit
template C10_HOST_DEVICE void blocksignificant<48, uint32_t>::setbit(size_t index, bool v) noexcept;
#pragma pop_macro("setbit")
template C10_HOST_DEVICE blocksignificant<48, uint32_t>& blocksignificant<48, uint32_t>::operator<<=(int bitsToShift);
template C10_HOST_DEVICE bool blocksignificant<48, uint32_t>::roundingDirection(size_t targetLsb) const;

#pragma diag_default 20040

}
}
