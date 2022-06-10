#pragma once

// Do not include this file manually, include UniversalTypes.h instead

#pragma push_macro("setbit")
#undef setbit
#include <universal/number/cfloat/cfloat.hpp>
#pragma pop_macro("setbit")

namespace sw {
namespace universal {

using CFloat = cfloat<32, 8, uint32_t, true, false, false>;
using BlockBinary = blockbinary<8, uint32_t, BinaryNumberType::Signed>;
using BlockTripleAdd = blocktriple<CFloat::fbits, BlockTripleOperator::ADD, uint32_t>;
using BlockTripleMul = blocktriple<CFloat::fbits, BlockTripleOperator::MUL, uint32_t>;
using BlockTripleDiv = blocktriple<CFloat::fbits, BlockTripleOperator::DIV, uint32_t>;
using BlockSignificantAdd = blocksignificant<BlockTripleAdd::bfbits, uint32_t>;
using BlockSignificantMul = blocksignificant<BlockTripleMul::bfbits, uint32_t>;
using BlockSignificantDiv = blocksignificant<BlockTripleDiv::bfbits, uint32_t>;

}
}
