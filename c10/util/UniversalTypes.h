#pragma once

#include <universal/number/cfloat/cfloat.hpp>

namespace c10 {

using CFloatWithSubnormals = sw::universal::cfloat<32, 8, uint32_t, true, false, false>;

inline bool operator<(double left, CFloatWithSubnormals right)
{
  return left < static_cast<double>(right);
}

inline bool operator>(double left, CFloatWithSubnormals right)
{
  return left > static_cast<double>(right);
}

}
