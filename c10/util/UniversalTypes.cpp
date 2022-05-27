#include <c10/util/UniversalTypes.h>

#include <limits>

namespace sw {
namespace universal {

template class cfloat<32, 8, uint32_t, true, false, false>;

// Suppress the warnings that __host__ functions were redeclared as __host__ __device__
#pragma diag_suppress 20040

// Instantiate cfloat methods and make them __host__ __device__
template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>::cfloat() noexcept;

template C10_HOST_DEVICE cfloat<32, 8, uint32_t, true, false, false>& cfloat<32, 8, uint32_t, true, false, false>::convert_ieee754<float>(float rhs) noexcept;

#pragma diag_default 20040

}
}

namespace std {

template class numeric_limits<c10::CFloatWithSubnormals>;

};
