#include <c10/util/UniversalTypes.h>

#include <limits>

namespace sw {
namespace universal {

template class cfloat<32, 8, uint32_t, true, false, false>;

}
}

namespace std {

template class numeric_limits<c10::CFloatWithSubnormals>;

};
