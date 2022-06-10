#pragma once

#pragma push_macro("setbit")
#undef setbit
#include <universal/number/cfloat/cfloat.hpp>
#pragma pop_macro("setbit")

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <type_traits>

#include <c10/util/UniversalTypes-CFloatWithSubnormals-add.h>
#include <c10/util/UniversalTypes-CFloatWithSubnormals-mul.h>
#include <c10/util/UniversalTypes-CFloatWithSubnormals-div.h>

// Private macro (it is removed with #undef below)
#define FORALL_SUPPORTED_TYPES(_) \
  _(int)                          \
  _(long)                         \
  _(long long)                    \
  _(float)                        \
  _(double)

  // Private macro (it is removed with #undef below)
  #define FORALL_SUPPORTED_TYPES_EXCEPT_DOUBLE(_) \
    _(int)                                        \
    _(long)                                       \
    _(long long)                                  \
    _(float)

// Private macro (it is removed with #undef below)
// (argument type, return type)
#define FORALL_SUPPORTED_TYPES_IN_OPERATORS(_) \
  _(int, CFloatWithSubnormals)                 \
  _(float, CFloatWithSubnormals)               \
  _(double, double) // ATen requires returning double if the argument type is double

// Private macro (it is removed with #undef below)
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
inline C10_HOST_DEVICE bool operator==(
  const cfloat<nnbits,nes,nbt,nsub,nsup,nsat>& lhs,
  const cfloat<nnbits,nes,nbt,nsub,nsup,nsat>& rhs);
template<size_t nnbits, size_t nes, typename nbt, bool nsub, bool nsup, bool nsat>
inline C10_HOST_DEVICE bool operator!=(
  const cfloat<nnbits,nes,nbt,nsub,nsup,nsat>& lhs,
  const cfloat<nnbits,nes,nbt,nsub,nsup,nsat>& rhs);

#define OP(T, _)                                                                                               \
  template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating>  \
  inline C10_HOST_DEVICE bool operator==(                                                                      \
    const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& lhs,                            \
    T rhs);                                                                                                    \
  template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating>  \
  inline C10_HOST_DEVICE bool operator!=(                                                                      \
    const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& lhs,                            \
    T rhs);
FORALL_SUPPORTED_TYPES_IN_OPERATORS(OP)
#undef OP

template<size_t nnbits, size_t nes, typename nbt, bool nsub, bool nsup, bool nsat>
inline C10_HOST_DEVICE bool operator<(
  const cfloat<nnbits, nes, nbt, nsub, nsup, nsat>& lhs,
  const cfloat<nnbits, nes, nbt, nsub, nsup, nsat>& rhs);

template<size_t nnbits, size_t nes, typename nbt, bool nsub, bool nsup, bool nsat>
inline C10_HOST_DEVICE bool operator<=(
  const cfloat<nnbits, nes, nbt, nsub, nsup, nsat>& lhs,
  const cfloat<nnbits, nes, nbt, nsub, nsup, nsat>& rhs);

template<size_t nnbits, size_t nes, typename nbt, bool nsub, bool nsup, bool nsat>
inline C10_HOST_DEVICE bool operator>(
  const cfloat<nnbits, nes, nbt, nsub, nsup, nsat>& lhs,
  const cfloat<nnbits, nes, nbt, nsub, nsup, nsat>& rhs);

template<size_t nnbits, size_t nes, typename nbt, bool nsub, bool nsup, bool nsat>
inline C10_HOST_DEVICE bool operator>=(
  const cfloat<nnbits, nes, nbt, nsub, nsup, nsat>& lhs,
  const cfloat<nnbits, nes, nbt, nsub, nsup, nsat>& rhs);

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

// Other math functions
template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating>
C10_HOST_DEVICE cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> exp(
  cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> x);

template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating>
C10_HOST_DEVICE cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> log(
  cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> x);

template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating>
C10_HOST_DEVICE cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> pow(
  cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> x,
  cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> y);

template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating>
C10_HOST_DEVICE cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> pow(
  cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> x,
  int y);

template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating>
C10_HOST_DEVICE cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> pow(
  cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> x,
  double y);


#define OP(T, _)                                                                                              \
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

// min and max
template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating>
C10_HOST_DEVICE cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> min(
  cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> x,
  cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> y);

template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating>
C10_HOST_DEVICE cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> max(
  cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> x,
  cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating> y);

// cfloat constructor
extern template C10_HOST_DEVICE CFloat::cfloat() noexcept;
extern template C10_HOST_DEVICE CFloat::cfloat(float) noexcept;
extern template C10_HOST_DEVICE CFloat::cfloat(double) noexcept;
extern template C10_HOST_DEVICE void CFloat::setblock(size_t b, uint32_t data) noexcept;

// blockbinary constructor and methods
extern template C10_HOST_DEVICE BlockBinary::blockbinary() noexcept;
extern template C10_HOST_DEVICE bool BlockBinary::isallones() const noexcept;
extern template C10_HOST_DEVICE bool BlockBinary::iszero() const noexcept;
extern template C10_HOST_DEVICE void BlockBinary::clear() noexcept;
extern template C10_HOST_DEVICE void BlockBinary::setbits(uint64_t value) noexcept;
#pragma push_macro("setbit")
#undef setbit
extern template C10_HOST_DEVICE void BlockBinary::setbit(size_t i, bool v) noexcept;
#pragma pop_macro("setbit")

// cfloat methods that handle blockbinary
extern template C10_HOST_DEVICE void CFloat::exponent(
  BlockBinary& e) const;

// isnan, isinf, and necessary methods
template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating>
inline C10_HOST_DEVICE bool isnan(const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& a);

template<size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating>
inline C10_HOST_DEVICE bool isinf(const cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& a);

extern template C10_HOST_DEVICE bool CFloat::isnan(int NaNType) const noexcept;
extern template C10_HOST_DEVICE bool CFloat::isnanencoding(int NaNType) const noexcept;
extern template C10_HOST_DEVICE bool CFloat::issupernormal() const noexcept;
extern template C10_HOST_DEVICE bool CFloat::isinf(int InfType) const noexcept;
extern template C10_HOST_DEVICE bool CFloat::ispos() const noexcept;
extern template C10_HOST_DEVICE bool CFloat::isneg() const noexcept;
extern template C10_HOST_DEVICE bool CFloat::sign() const noexcept;

// Conversion to float and necessary methods
extern template C10_HOST_DEVICE CFloat::operator float() const noexcept;
extern template C10_HOST_DEVICE bool CFloat::iszero() const noexcept;
extern template C10_HOST_DEVICE bool CFloat::iszeroencoding() const noexcept;
extern template C10_HOST_DEVICE bool CFloat::at(size_t bitIndex) const noexcept;

// Conversion from float and necessary methods
extern template C10_HOST_DEVICE CFloat&
  CFloat::convert_ieee754<float>(float rhs) noexcept;
extern template C10_HOST_DEVICE void CFloat::setnan(int NaNType) noexcept;
extern template C10_HOST_DEVICE void CFloat::setinf(bool sign) noexcept;
extern template C10_HOST_DEVICE void CFloat::clear() noexcept;
extern template C10_HOST_DEVICE CFloat& CFloat::flip() noexcept;
extern template C10_HOST_DEVICE CFloat& CFloat::maxneg() noexcept;
extern template C10_HOST_DEVICE CFloat& CFloat::maxpos() noexcept;
extern template C10_HOST_DEVICE void CFloat::setsign(bool sign);
extern template C10_HOST_DEVICE void CFloat::shiftLeft(int leftShift);
extern template C10_HOST_DEVICE void CFloat::shiftRight(int rightShift);

// Explicit type casts to types other than float
extern template C10_HOST_DEVICE CFloat::operator int() const noexcept;
extern template C10_HOST_DEVICE CFloat::operator long() const noexcept;
extern template C10_HOST_DEVICE CFloat::operator long long() const noexcept;
extern template C10_HOST_DEVICE CFloat::operator double() const noexcept;
extern template C10_HOST_DEVICE int CFloat::to_int() const;
extern template C10_HOST_DEVICE long CFloat::to_long() const;
extern template C10_HOST_DEVICE long long CFloat::to_long_long() const;

// extractFields and necessary functions (required for conversion from float and from double)
inline C10_HOST_DEVICE void extractFields(float value, bool& s, uint64_t& rawExponentBits, uint64_t& rawFractionBits);

// operator=
extern template C10_HOST_DEVICE CFloat& CFloat::operator=(
  float rhs) noexcept;

// to_native
// FIXME extern template causes "Undefined reference" error. Fix it and move instantiation to UniversalTypes.cpp
template C10_HOST_DEVICE float CFloat::to_native<float>() const;
template C10_HOST_DEVICE double CFloat::to_native<double>() const;

#pragma diag_default 20040

}
}

namespace c10 {

static constexpr C10_DEVICE int subnormal_reciprocal_shift_device[] = {
	0,                    // es =  0 : not a valid value
	-1,                   // es =  1 : 2^(2 - 2^(es-1)) = 2^1
	0,                    // es =  2 : 2^(2 - 2^(es-1)) = 2^0
	2,                    // es =  3 : 2^(2 - 2^(es-1)) = 2^-2
	6,                    // es =  4 : 2^(2 - 2^(es-1)) = 2^-6
	14,                   // es =  5 : 2^(2 - 2^(es-1)) = 2^-14
	30,                   // es =  6 : 2^(2 - 2^(es-1)) = 2^-30
	62,                   // es =  7 : 2^(2 - 2^(es-1)) = 2^-62
	126,                  // es =  8 : 2^(2 - 2^(es-1)) = 2^-126
	254,                  // es =  9 : 2^(2 - 2^(es-1)) = 2^-254
	510,                  // es = 10 : 2^(2 - 2^(es-1)) = 2^-510
	1022,                 // es = 11 : 2^(2 - 2^(es-1)) = 2^-1022
	2046,                 // es = 12 : 2^(2 - 2^(es-1)) = 2^-2046
	4094,                 // es = 13 : 2^(2 - 2^(es-1)) = 2^-4094
	8190,                 // es = 14 : 2^(2 - 2^(es-1)) = 2^-8190
	16382,                // es = 15 : 2^(2 - 2^(es-1)) = 2^-16382
	32766,                // es = 16 : 2^(2 - 2^(es-1)) = 2^-32766
	65534,                // es = 17 : 2^(2 - 2^(es-1)) = 2^-65534
	131070,               // es = 18 : 2^(2 - 2^(es-1)) = 2^-131070
	262142,               // es = 19 : 2^(2 - 2^(es-1)) = 2^-262142
	524286                // es = 20 : 2^(2 - 2^(es-1)) = 2^-524286
};

// __host__ __device__ version of convert
template<
  size_t srcbits, sw::universal::BlockTripleOperator op,
  size_t nbits, size_t es, typename bt, bool hasSubnormals, bool hasSupernormals, bool isSaturating>
inline C10_HOST_DEVICE void convert(
    const sw::universal::blocktriple<srcbits, op, bt>& src,
    sw::universal::cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>& tgt)
{
  using btType = sw::universal::blocktriple<srcbits, op, bt>;
	using cfloatType = sw::universal::cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>;
	// test special cases
	if (src.isnan()) {
		tgt.setnan(src.sign() ? sw::universal::NAN_TYPE_SIGNALLING : sw::universal::NAN_TYPE_QUIET);
	}
	else	if (src.isinf()) {
		tgt.setinf(src.sign());
	}
	else 	if (src.iszero()) {
		tgt.setzero();
		tgt.setsign(src.sign()); // preserve sign
	}
	else {
		int significantScale = src.significantscale();
		int exponent = src.scale() + significantScale;

    #ifdef __CUDA_ARCH__
    constexpr int subnormal_reciprocal_shift = subnormal_reciprocal_shift_device[es];
    #else
    constexpr int subnormal_reciprocal_shift = sw::universal::subnormal_reciprocal_shift[es];
    #endif

		// special case of underflow
		if (hasSubnormals) {
//			std::cout << "exponent = " << exponent << " bias = " << cfloatType::EXP_BIAS << " exp subnormal = " << cfloatType::MIN_EXP_SUBNORMAL << '\n';
			// why must exponent be less than (minExpSubnormal - 1) to be rounded to zero?
			// because the half-way value that would round up to minpos is at exp = (minExpSubnormal - 1)
			if (exponent < cfloatType::MIN_EXP_SUBNORMAL) {
				tgt.setzero();
				if (exponent == (cfloatType::MIN_EXP_SUBNORMAL - 1)) {
					// -exponent because we are right shifting and exponent in this range is negative
					int adjustment = -(exponent + subnormal_reciprocal_shift);
					std::pair<bool, size_t> alignment = src.roundingDecision(adjustment);
					if (alignment.first) ++tgt; // we are minpos
				}
				tgt.setsign(src.sign());
				return;
			}
		}
		else {
			if (exponent + cfloatType::EXP_BIAS <= 0) {  // value is in the subnormal range, which maps to 0
				tgt.setzero();
				tgt.setsign(src.sign());
				return;
			}
		}
		// special case of overflow
		if (hasSupernormals) {
			if (isSaturating) {
				if (exponent > cfloatType::MAX_EXP) {
					if (src.sign()) tgt.maxneg(); else tgt.maxpos();
					return;
				}
			}
			else {
				if (exponent > cfloatType::MAX_EXP) {
					tgt.setinf(src.sign());
					return;
				}
			}
		}
		else {  // no supernormals will saturate at a different encoding: TODO can we hide it all in maxpos?
			if (isSaturating) {
				if (exponent > cfloatType::MAX_EXP) {
					if (src.sign()) tgt.maxneg(); else tgt.maxpos();
					return;
				}
			}
			else {
				if (exponent > cfloatType::MAX_EXP) {
					tgt.setinf(src.sign());
					return;
				}
			}
		}

		// our value needs to go through rounding to be correctly interpreted
		//
		// tgt.clear();  // no need as all bits are going to be set by the code below

		// exponent construction
		int adjustment{ 0 };
		// construct exponent
		uint64_t biasedExponent = static_cast<uint64_t>(static_cast<long long>(exponent) + static_cast<long long>(cfloatType::EXP_BIAS)); // this is guaranteed to be positive if exponent in encoding range
//			std::cout << "exponent         " << to_binary(biasedExponent) << '\n';
		if (hasSubnormals) {
			//if (exponent >= cfloatType::MIN_EXP_SUBNORMAL && exponent < cfloat<nbits, es, bt, hasSubnormals, hasSupernormals, isSaturating>::MIN_EXP_NORMAL) {
			if (exponent < cfloatType::MIN_EXP_NORMAL) {
				// the value is in the subnormal range of the cfloat
				biasedExponent = 0;
				// -exponent because we are right shifting and exponent in this range is negative
				adjustment = -(exponent + subnormal_reciprocal_shift);
				// this is the right shift adjustment required for subnormal representation due
				// to the scale of the input number, i.e. the exponent of 2^-adjustment
			}
			else {
				// the value is in the normal range of the cfloat
				biasedExponent = static_cast<uint64_t>(static_cast<long long>(exponent) + static_cast<long long>(cfloatType::EXP_BIAS)); // this is guaranteed to be positive
			}
		}
		else {
			if (exponent < cfloatType::MIN_EXP_NORMAL) biasedExponent = 1ull; // fixup biasedExponent if we are in the subnormal region
		}


		// get the rounding direction and the LSB right shift:
		std::pair<bool, size_t> alignment = src.roundingDecision(adjustment);
		bool roundup = alignment.first;
		size_t rightShift = alignment.second;  // this is the shift to get the LSB of the src to the LSB of the tgt
		//std::cout << "round-up?        " << (roundup ? "yes" : "no") << '\n';
		//std::cout << "rightShift       " << rightShift << '\n';

		if (btType::bfbits < 65) {
			// we can use a uint64_t to construct the cfloat
			uint64_t raw = (src.sign() ? 1ull : 0ull); // process sign
			//std::cout << "raw bits (sign)  " << to_binary(raw) << '\n';
			// construct the fraction bits
			uint64_t fracbits = src.significant_ull(); // get all the bits, including the integer bits
			//std::cout << "fracbits         " << to_binary(fracbits) << '\n';
			fracbits >>= rightShift;
			//std::cout << "fracbits shifted " << to_binary(fracbits) << '\n';
			fracbits &= cfloatType::ALL_ONES_FR; // remove the hidden bit
			//std::cout << "fracbits masked  " << to_binary(fracbits) << '\n';
			if (roundup) ++fracbits;
			if (fracbits == (1ull << cfloatType::fbits)) { // check for overflow
				if (biasedExponent == cfloatType::ALL_ONES_ES) {
					fracbits = cfloatType::INF_ENCODING; // project to INF
				}
				else {
					++biasedExponent;
					fracbits = 0;
				}
			}

			raw <<= es; // shift sign to make room for the exponent bits
			raw |= biasedExponent;
			//std::cout << "raw bits (exp)   " << to_binary(raw) << '\n';
			raw <<= cfloatType::fbits; // make room for the fraction bits
			//std::cout << "raw bits (s+exp) " << to_binary(raw) << '\n';
			raw |= fracbits;
			//std::cout << "raw bits (final) " << to_binary(raw) << '\n';
			tgt.setbits(raw);
//			std::cout << "raw bits (all)   " << to_binary(raw) << '\n';
			// when you get too far, map it back to +-inf: TBD: this doesn't appear to be the right algorithm to catch all overflow patterns
			if (tgt.isnan()) tgt.setinf(src.sign());	// map back to +-inf
		}
		else {
			// compose the segments
			auto fracbits = src.significant();  // why significant? cheesy optimization: we are going to overwrite the hidden bit position anyway when we write the exponent below, so no need to pay the overhead of generating the fraction here.
			//std::cout << "fraction      : " << to_binary(fracbits, true) << '\n';
			fracbits >>= static_cast<int>(rightShift);
			//std::cout << "aligned fbits : " << to_binary(fracbits, true) << '\n';

			// copy the blocks that contain fraction bits
			// significant blocks are organized like this:
			//   ADD        iii.ffffrrrrrrrrr          3 integer bits, f fraction bits, and 2*fhbits rounding bits
			//   MUL         ii.ffff'ffff              2 integer bits, 2*f fraction bits
			//   DIV         ii.ffff'ffff'ffff'rrrr    2 integer bits, 3*f fraction bits, and r rounding bits
			//std::cout << "fraction bits : " << to_binary(fracbits, true) << '\n';
			tgt.clear();
			//std::cout << "initial state : " << to_binary(tgt) << " : " << tgt << '\n';
			for (size_t b = 0; b < btType::nrBlocks; ++b) {
				tgt.setblock(b, fracbits.block(b));
			}
			//std::cout << "fraction bits : " << to_binary(tgt, true) << '\n';
			tgt.setsign(src.sign());
			//std::cout << "adding sign   : " << to_binary(tgt) << '\n';
      tgt.setexponent(exponent);
			// if (!tgt.setexponent(exponent)) {
			// 	std::cerr << "exponent value is out of range: " << exponent << '\n';
			// }
			//std::cout << "add exponent  : " << to_binary(tgt) << '\n';
		}
	}
}


class alignas(4) CFloatWithSubnormals : public sw::universal::cfloat<32, 8, uint32_t, true, false, false>
{
public:
  using Base = sw::universal::cfloat<32, 8, uint32_t, true, false, false>;

  constexpr C10_HOST_DEVICE CFloatWithSubnormals() = default;

  C10_HOST_DEVICE CFloatWithSubnormals(float value) : Base()
  {
    convert_ieee754<float>(value);
  }

  constexpr C10_HOST_DEVICE CFloatWithSubnormals(Base value) : Base(value) {}

  // A way to construct CFloatWithSubnormals from bits similar to Half and BFloat16
  struct from_bits_t {};
  static constexpr C10_HOST_DEVICE from_bits_t from_bits()
  {
    return from_bits_t();
  }

  constexpr C10_HOST_DEVICE CFloatWithSubnormals(uint32_t bits, from_bits_t)
  {
    setblock(0, bits);
  }

  C10_HOST_DEVICE operator float() const noexcept
  {
    return Base::operator float();
  }

  // Define operator < to avoid ambiguity
  C10_HOST_DEVICE bool operator<(const CFloatWithSubnormals& right) const
  {
    return sw::universal::operator<(
      static_cast<Base>(*this),
      static_cast<Base>(right));
  }
  C10_HOST_DEVICE bool operator<=(const CFloatWithSubnormals& right) const
  {
    return sw::universal::operator<=(
      static_cast<Base>(*this),
      static_cast<Base>(right));
  }
  #define OP(T)                                    \
    C10_HOST_DEVICE bool operator<(T right) const  \
    {                                              \
      return sw::universal::operator<(             \
        static_cast<Base>(*this),                  \
        static_cast<Base>(right));                 \
    }                                              \
    C10_HOST_DEVICE bool operator<=(T right) const \
    {                                              \
      return sw::universal::operator<=(            \
        static_cast<Base>(*this),                  \
        static_cast<Base>(right));                 \
    }
  FORALL_SUPPORTED_TYPES(OP)
  FORALL_ADDITIONAL_TYPES(OP)
  #undef OP

  // Define operator > to avoid ambiguity
  C10_HOST_DEVICE bool operator>(const CFloatWithSubnormals& right) const
  {
    return sw::universal::operator>(
      static_cast<Base>(*this),
      static_cast<Base>(right));
  }
  C10_HOST_DEVICE bool operator>=(const CFloatWithSubnormals& right) const
  {
    return sw::universal::operator>=(
      static_cast<Base>(*this),
      static_cast<Base>(right));
  }
  #define OP(T)                                    \
    C10_HOST_DEVICE bool operator>(T right) const  \
    {                                              \
      return sw::universal::operator>(             \
        static_cast<Base>(*this),                  \
        static_cast<Base>(right));                 \
    }                                              \
    C10_HOST_DEVICE bool operator>=(T right) const \
    {                                              \
      return sw::universal::operator>=(            \
        static_cast<Base>(*this),                  \
        static_cast<Base>(right));                 \
    }
  FORALL_SUPPORTED_TYPES(OP)
  FORALL_ADDITIONAL_TYPES(OP)
  #undef OP

  // Assignment operators
  #define OP(T)                                              \
    C10_HOST_DEVICE CFloatWithSubnormals& operator=(T value) \
    {                                                        \
      Base::operator=(value);                                \
      return *this;                                          \
    }
  FORALL_SUPPORTED_TYPES_EXCEPT_DOUBLE(OP)
  #undef OP
  // FIXME remove this once std::cout gets removed from the double conversion
  C10_HOST_DEVICE CFloatWithSubnormals& operator=(double value)
  {
    Base::operator=(static_cast<float>(value));
    return *this;
  }
  C10_HOST_DEVICE CFloatWithSubnormals& operator=(signed char value)
  {
    convert_signed_integer(value);
    return *this;
  }
  C10_HOST_DEVICE CFloatWithSubnormals& operator=(unsigned char value)
  {
    convert_unsigned_integer(value);
    return *this;
  }
  C10_HOST_DEVICE CFloatWithSubnormals& operator=(short value)
  {
    convert_signed_integer(value);
    return *this;
  }
  C10_HOST_DEVICE CFloatWithSubnormals& operator=(bool value)
  {
    convert_signed_integer(value);
    return *this;
  }
  C10_HOST_DEVICE CFloatWithSubnormals& operator=(c10::BFloat16 value)
  {
    Base::operator=(static_cast<float>(value));
    return *this;
  }
  C10_HOST_DEVICE CFloatWithSubnormals& operator=(c10::Half value)
  {
    Base::operator=(static_cast<float>(value));
    return *this;
  }

  // Arithmetic operators
  // ATen requires unary minus to be constexpr
  constexpr C10_HOST_DEVICE CFloatWithSubnormals operator-() const
  {
    cfloat tmp(*this);
    tmp.setblock(Base::MSU, tmp.block(Base::MSU) ^ Base::SIGN_BIT_MASK);
    return static_cast<CFloatWithSubnormals>(tmp);
  }
  C10_HOST_DEVICE CFloatWithSubnormals& operator+=(const CFloatWithSubnormals& rhs)
  {
    // This operator has to use the redefined convert
		// special case handling of the inputs
#if CFLOAT_THROW_ARITHMETIC_EXCEPTION
		if (isnan(sw::universal::NAN_TYPE_SIGNALLING) || rhs.isnan(sw::universal::NAN_TYPE_SIGNALLING)) {
			throw sw::universal::cfloat_operand_is_nan{};
		}
#else
		if (isnan(sw::universal::NAN_TYPE_SIGNALLING) || rhs.isnan(sw::universal::NAN_TYPE_SIGNALLING)) {
			setnan(sw::universal::NAN_TYPE_SIGNALLING);
			return *this;
		}
		if (isnan(sw::universal::NAN_TYPE_QUIET) || rhs.isnan(sw::universal::NAN_TYPE_QUIET)) {
			setnan(sw::universal::NAN_TYPE_QUIET);
			return *this;
		}
#endif
		// normal + inf  = inf
		// normal + -inf = -inf
		// inf + normal = inf
		// inf + inf    = inf
		// inf + -inf    = ?
		// -inf + normal = -inf
		// -inf + -inf   = -inf
		// -inf + inf    = ?
		if (isinf()) {
			if (rhs.isinf()) {
				if (sign() != rhs.sign()) {
					setnan(sw::universal::NAN_TYPE_SIGNALLING);
				}
				return *this;
			}
			else {
				return *this;
			}
		}
		else {
			if (rhs.isinf()) {
				*this = rhs;
				return *this;
			}
		}

		if (iszero()) {
			*this = rhs;
			return *this;
		}
		if (rhs.iszero()) return *this;

		// arithmetic operation
		sw::universal::blocktriple<fbits, sw::universal::BlockTripleOperator::ADD, uint32_t> a, b, sum;

		// transform the inputs into (sign,scale,significant)
		// triples of the correct width
		normalizeAddition(a);
		rhs.normalizeAddition(b);
		sum.add(a, b);

		c10::convert(sum, *this);

		return *this;
  }
  C10_HOST_DEVICE CFloatWithSubnormals& operator-=(const CFloatWithSubnormals& rhs)
  {
    // This operator has to use the redefined convert
		if (rhs.isnan())
			return *this += rhs;
		else
			return *this += -rhs;
  }
  C10_HOST_DEVICE CFloatWithSubnormals& operator*=(const CFloatWithSubnormals& rhs)
  {
    // This operator has to use the redefined convert
		// special case handling of the inputs
#if CFLOAT_THROW_ARITHMETIC_EXCEPTION
		if (isnan(sw::universal::NAN_TYPE_SIGNALLING) || rhs.isnan(sw::universal::NAN_TYPE_SIGNALLING)) {
			throw cfloat_operand_is_nan{};
		}
#else
		if (isnan(sw::universal::NAN_TYPE_SIGNALLING) || rhs.isnan(sw::universal::NAN_TYPE_SIGNALLING)) {
			setnan(sw::universal::NAN_TYPE_SIGNALLING);
			return *this;
		}
		if (isnan(sw::universal::NAN_TYPE_QUIET) || rhs.isnan(sw::universal::NAN_TYPE_QUIET)) {
			setnan(sw::universal::NAN_TYPE_QUIET);
			return *this;
		}
#endif
		//  inf * inf = inf
		//  inf * -inf = -inf
		// -inf * inf = -inf
		// -inf * -inf = inf
		//	0 * inf = -nan(ind)
		bool resultSign = sign() != rhs.sign();
		if (isinf()) {
			if (rhs.isinf()) {
				setsign(resultSign);
				return *this;
			}
			else {
				setnan(sw::universal::NAN_TYPE_SIGNALLING);
				return *this;
			}
		}
		else {
			if (rhs.isinf()) {
				setnan(sw::universal::NAN_TYPE_SIGNALLING);
				return *this;
			}
		}

		if (iszero() || rhs.iszero()) {
			setzero();
			setsign(resultSign); // deal with negative 0
			return *this;
		}

		// arithmetic operation
		sw::universal::blocktriple<fbits, sw::universal::BlockTripleOperator::MUL, uint32_t> a, b, product;

		// transform the inputs into (sign,scale,significant)
		// triples of the correct width
		normalizeMultiplication(a);
		rhs.normalizeMultiplication(b);
		product.mul(a, b);
		c10::convert(product, *this);

		return *this;
  }
  C10_HOST_DEVICE CFloatWithSubnormals& operator/=(const CFloatWithSubnormals& rhs)
  {
    // special case handling of the inputs
		// qnan / qnan = qnan
		// qnan / snan = qnan
		// snan / qnan = snan
		// snan / snan = snan
#if CFLOAT_THROW_ARITHMETIC_EXCEPTION
		if (rhs.iszero()) throw sw::universal::cfloat_divide_by_zero();
		if (rhs.isnan()) throw sw::universal::cfloat_divide_by_nan();
		if (isnan()) throw sw::universal::cfloat_operand_is_nan();
#else
		if (isnan(sw::universal::NAN_TYPE_SIGNALLING) || rhs.isnan(sw::universal::NAN_TYPE_SIGNALLING)) {
			setnan(sw::universal::NAN_TYPE_SIGNALLING);
			return *this;
		}
		if (isnan(sw::universal::NAN_TYPE_QUIET) || rhs.isnan(sw::universal::NAN_TYPE_QUIET)) {
			setnan(sw::universal::NAN_TYPE_QUIET);
			return *this;
		}
		if (rhs.iszero()) {
			if (iszero()) {
				// zero divide by zero yields quiet NaN (in MSVC it is labeled -nan(ind) for indeterminate)
				setnan(sw::universal::NAN_TYPE_QUIET);
			}
			else {
				// non-zero divide by zero yields INF
				bool resultSign = sign() != rhs.sign();
				setinf(resultSign);
			}
			return *this;
		}
#endif
		//  inf /  inf = -nan(ind)
		//  inf / -inf = -nan(ind)
		// -inf /  inf = -nan(ind)
		// -inf / -inf = -nan(ind)
		//	1.0 /  inf = 0
		bool resultSign = sign() != rhs.sign();
		if (isinf()) {
			if (rhs.isinf()) {
				// inf divide by inf yields quiet NaN (in MSVC it is labeled -nan(ind) for indeterminate)
				setnan(sw::universal::NAN_TYPE_QUIET);
				return *this;
			}
			else {
				// we stay an infinite but may change sign
				setsign(resultSign);
				return *this;
			}
		}
		else {
			if (rhs.isinf()) {
				setzero();
				setsign(resultSign);
				return *this;
			}
		}

		if (iszero()) {
			setzero();
			setsign(resultSign); // deal with negative 0
			return *this;
		}

		// arithmetic operation
		using BlockTriple = sw::universal::blocktriple<fbits, sw::universal::BlockTripleOperator::DIV, uint32_t>;
		BlockTriple a, b, quotient;

		// transform the inputs into (sign,scale,significant)
		// triples of the correct width
		normalizeDivision(a);
		rhs.normalizeDivision(b);
		quotient.div(a, b);
		quotient.setradix(BlockTriple::radix);
		c10::convert(quotient, *this);

		return *this;
  }
};

// Nonmember comparison operators
#define OP(T)                                                                       \
  inline C10_HOST_DEVICE bool operator<(T left, const CFloatWithSubnormals& right)  \
  {                                                                                 \
    return sw::universal::operator<(                                                \
      static_cast<CFloatWithSubnormals::Base>(left),                                \
      static_cast<CFloatWithSubnormals::Base>(right));                              \
  }                                                                                 \
  inline C10_HOST_DEVICE bool operator<=(T left, const CFloatWithSubnormals& right) \
  {                                                                                 \
    return sw::universal::operator<=(                                               \
      static_cast<CFloatWithSubnormals::Base>(left),                                \
      static_cast<CFloatWithSubnormals::Base>(right));                              \
  }
FORALL_SUPPORTED_TYPES(OP)
FORALL_ADDITIONAL_TYPES(OP)
#undef OP

#define OP(T)                                                                       \
  inline C10_HOST_DEVICE bool operator>(T left, const CFloatWithSubnormals& right)  \
  {                                                                                 \
    return sw::universal::operator>(                                                \
      static_cast<CFloatWithSubnormals::Base>(left),                                \
      static_cast<CFloatWithSubnormals::Base>(right));                              \
  }                                                                                 \
  inline C10_HOST_DEVICE bool operator>=(T left, const CFloatWithSubnormals& right) \
  {                                                                                 \
    return sw::universal::operator>=(                                               \
      static_cast<CFloatWithSubnormals::Base>(left),                                \
      static_cast<CFloatWithSubnormals::Base>(right));                              \
  }
FORALL_SUPPORTED_TYPES(OP)
FORALL_ADDITIONAL_TYPES(OP)
#undef OP

// Nonmember arithmetic operators
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

#define OP(T, R)                                                                                                                 \
  inline C10_HOST_DEVICE R operator+(const CFloatWithSubnormals& left, T right)                            \
  {                                                                                                                           \
    return static_cast<R>(sw::universal::operator+(static_cast<CFloatWithSubnormals::Base>(left), right)); \
  }                                                                                                                           \
  inline C10_HOST_DEVICE R operator-(const CFloatWithSubnormals& left, T right)                            \
  {                                                                                                                           \
    return static_cast<R>(sw::universal::operator-(static_cast<CFloatWithSubnormals::Base>(left), right)); \
  }                                                                                                                           \
  inline C10_HOST_DEVICE R operator*(const CFloatWithSubnormals& left, T right)                            \
  {                                                                                                                           \
    return static_cast<R>(sw::universal::operator*(static_cast<CFloatWithSubnormals::Base>(left), right)); \
  }                                                                                                                           \
  inline C10_HOST_DEVICE R operator/(const CFloatWithSubnormals& left, T right)                            \
  {                                                                                                                           \
    return static_cast<R>(sw::universal::operator/(static_cast<CFloatWithSubnormals::Base>(left), right)); \
  }                                                                                                                           \
  inline C10_HOST_DEVICE R operator+(T left, const CFloatWithSubnormals& right)                            \
  {                                                                                                                           \
    return static_cast<R>(sw::universal::operator+(left, static_cast<CFloatWithSubnormals::Base>(right))); \
  }                                                                                                                           \
  inline C10_HOST_DEVICE R operator-(T left, const CFloatWithSubnormals& right)                            \
  {                                                                                                                           \
    return static_cast<R>(sw::universal::operator-(left, static_cast<CFloatWithSubnormals::Base>(right))); \
  }                                                                                                                           \
  inline C10_HOST_DEVICE R operator*(T left, const CFloatWithSubnormals& right)                            \
  {                                                                                                                           \
    return static_cast<R>(sw::universal::operator*(left, static_cast<CFloatWithSubnormals::Base>(right))); \
  }                                                                                                                           \
  inline C10_HOST_DEVICE R operator/(T left, const CFloatWithSubnormals& right)                            \
  {                                                                                                                           \
    return static_cast<R>(sw::universal::operator/(left, static_cast<CFloatWithSubnormals::Base>(right))); \
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
#undef FORALL_SUPPORTED_TYPES_EXCEPT_DOUBLE
#undef FORALL_SUPPORTED_TYPES_IN_OPERATORS
#undef FORALL_ADDITIONAL_TYPES

// Checks whether T is one of the new floating point types
template <typename T>
struct is_universal_floating_point :
  std::integral_constant<bool,
    std::is_same<T, CFloatWithSubnormals>::value> {
};

// In case we want to add complex or integral types later
template <typename T>
struct is_universal_type :
  std::integral_constant<bool,
    is_universal_floating_point<T>::value> {
};

}

namespace std {

template<>
struct hash<c10::CFloatWithSubnormals> {
  size_t operator()(const c10::CFloatWithSubnormals& value) const noexcept {
    return hash<uint32_t>()(value.block(0));
  }
};

template<> class numeric_limits<c10::CFloatWithSubnormals>
{
public:
  static constexpr bool is_specialized = true;
  static constexpr c10::CFloatWithSubnormals min() { // return minimum value
    return c10::CFloatWithSubnormals(numeric_limits<c10::CFloatWithSubnormals::Base>::min());
  }
  static constexpr C10_HOST_DEVICE c10::CFloatWithSubnormals max() { // return maximum value
    return c10::CFloatWithSubnormals(numeric_limits<c10::CFloatWithSubnormals::Base>::max());
  }
  static constexpr C10_HOST_DEVICE c10::CFloatWithSubnormals lowest() { // return most negative value
    return c10::CFloatWithSubnormals(numeric_limits<c10::CFloatWithSubnormals::Base>::lowest());
  }
  static constexpr c10::CFloatWithSubnormals epsilon() { // return smallest effective increment from 1.0
    return c10::CFloatWithSubnormals(numeric_limits<c10::CFloatWithSubnormals::Base>::epsilon());
  }
  static constexpr c10::CFloatWithSubnormals round_error() { // return largest rounding error
    return c10::CFloatWithSubnormals(numeric_limits<c10::CFloatWithSubnormals::Base>::round_error());
  }
  static constexpr c10::CFloatWithSubnormals denorm_min() {  // return minimum denormalized value
    return c10::CFloatWithSubnormals(numeric_limits<c10::CFloatWithSubnormals::Base>::denorm_min());
  }
  static constexpr C10_HOST_DEVICE c10::CFloatWithSubnormals infinity() { // return positive infinity
    return c10::CFloatWithSubnormals(numeric_limits<c10::CFloatWithSubnormals::Base>::infinity());
  }
  static constexpr c10::CFloatWithSubnormals quiet_NaN() { // return non-signaling NaN
    return c10::CFloatWithSubnormals(numeric_limits<c10::CFloatWithSubnormals::Base>::quiet_NaN());
  }
  static constexpr c10::CFloatWithSubnormals signaling_NaN() { // return signaling NaN
    return c10::CFloatWithSubnormals(numeric_limits<c10::CFloatWithSubnormals::Base>::signaling_NaN());
  }

  static constexpr int digits       = numeric_limits<c10::CFloatWithSubnormals::Base>::digits;
  static constexpr int digits10     = numeric_limits<c10::CFloatWithSubnormals::Base>::digits10;
  static constexpr int max_digits10 = numeric_limits<c10::CFloatWithSubnormals::Base>::max_digits10;
  static constexpr bool is_signed   = numeric_limits<c10::CFloatWithSubnormals::Base>::is_signed;
  static constexpr bool is_integer  = numeric_limits<c10::CFloatWithSubnormals::Base>::is_integer;
  static constexpr bool is_exact    = numeric_limits<c10::CFloatWithSubnormals::Base>::is_exact;
  static constexpr int radix        = numeric_limits<c10::CFloatWithSubnormals::Base>::radix;

  static constexpr int min_exponent   = numeric_limits<c10::CFloatWithSubnormals::Base>::min_exponent;
  static constexpr int min_exponent10 = numeric_limits<c10::CFloatWithSubnormals::Base>::min_exponent10;
  static constexpr int max_exponent   = numeric_limits<c10::CFloatWithSubnormals::Base>::max_exponent;
  static constexpr int max_exponent10 = numeric_limits<c10::CFloatWithSubnormals::Base>::max_exponent10;
  static constexpr bool has_infinity  = numeric_limits<c10::CFloatWithSubnormals::Base>::has_infinity;
  static constexpr bool has_quiet_NaN = numeric_limits<c10::CFloatWithSubnormals::Base>::has_quiet_NaN;
  static constexpr bool has_signaling_NaN = numeric_limits<c10::CFloatWithSubnormals::Base>::has_signaling_NaN;
  static constexpr float_denorm_style has_denorm = numeric_limits<c10::CFloatWithSubnormals::Base>::has_denorm;
  static constexpr bool has_denorm_loss = numeric_limits<c10::CFloatWithSubnormals::Base>::has_denorm_loss;

  static constexpr bool is_iec559 = numeric_limits<c10::CFloatWithSubnormals::Base>::is_iec559;
  static constexpr bool is_bounded = numeric_limits<c10::CFloatWithSubnormals::Base>::is_bounded;
  static constexpr bool is_modulo = numeric_limits<c10::CFloatWithSubnormals::Base>::is_modulo;
  static constexpr bool traps = numeric_limits<c10::CFloatWithSubnormals::Base>::traps;
  static constexpr bool tinyness_before = numeric_limits<c10::CFloatWithSubnormals::Base>::tinyness_before;
  static constexpr float_round_style round_style = numeric_limits<c10::CFloatWithSubnormals::Base>::round_style;
};

}
