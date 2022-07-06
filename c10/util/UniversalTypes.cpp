#include <c10/util/UniversalTypes.h>

#include <limits>

namespace sw {
namespace universal {

// Suppress the warnings that __host__ functions were redeclared as __host__ __device__
#pragma diag_suppress 20040

// blockbinary constructor and methods
template C10_HOST_DEVICE bool blockbinary<8, uint32_t, BinaryNumberType::Signed>::isallones() const noexcept;
template C10_HOST_DEVICE void blockbinary<8, uint32_t, BinaryNumberType::Signed>::clear() noexcept;
template C10_HOST_DEVICE void blockbinary<8, uint32_t, BinaryNumberType::Signed>::setbits(uint64_t value) noexcept;
#pragma push_macro("setbit")
#undef setbit
template C10_HOST_DEVICE void blockbinary<8, uint32_t, BinaryNumberType::Signed>::setbit(size_t i, bool v) noexcept;
#pragma pop_macro("setbit")

// cfloat methods that handle blockbinary
template C10_HOST_DEVICE void cfloat<32, 8, uint32_t, true, false, false>::exponent(
  blockbinary<8, uint32_t, BinaryNumberType::Signed>& e) const;

// isnan and necessary methods
template C10_HOST_DEVICE bool cfloat<32, 8, uint32_t, true, false, false>::isnan(int NaNType) const noexcept;
template C10_HOST_DEVICE bool cfloat<32, 8, uint32_t, true, false, false>::isnanencoding(int NaNType) const noexcept;
template C10_HOST_DEVICE bool cfloat<32, 8, uint32_t, true, false, false>::issupernormal() const noexcept;
template C10_HOST_DEVICE bool cfloat<32, 8, uint32_t, true, false, false>::isinf(int InfType) const noexcept;
template C10_HOST_DEVICE bool cfloat<32, 8, uint32_t, true, false, false>::ispos() const noexcept;
template C10_HOST_DEVICE bool cfloat<32, 8, uint32_t, true, false, false>::isneg() const noexcept;
template C10_HOST_DEVICE bool cfloat<32, 8, uint32_t, true, false, false>::sign() const noexcept;

#pragma diag_default 20040

}
}
