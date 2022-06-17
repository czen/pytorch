# flake8: noqa F401

"""The implementations should be moved here as soon as their deprecation period is over."""
from torch.testing._legacy import (
    _validate_dtypes,
    _dispatch_dtypes,
    _all_types_and_complex,
    all_types,
    all_types_and,
    all_types_and_complex,
    all_types_and_complex_and,
    all_types_and_half,
    complex_types,
    empty_types,
    floating_and_complex_types,
    floating_and_complex_types_and,
    floating_types,
    floating_types_and,
    double_types,
    floating_types_and_half,
    get_all_complex_dtypes,
    get_all_dtypes,
    get_all_fp_dtypes,
    get_all_int_dtypes,
    get_all_math_dtypes,
    integral_types,
    integral_types_and,
)
import torch

_universal_types = _dispatch_dtypes((torch.cfloatwithsubnormals,))
def universal_types():
    return _universal_types

_all_types_and_complex_and_universal = _all_types_and_complex + _universal_types
def all_types_and_complex_and_universal():
    return _all_types_and_complex_and_universal

def all_types_and_complex_and_universal_and(*dtypes):
    return _all_types_and_complex_and_universal + _validate_dtypes(*dtypes)

def get_all_dtypes_and_universal(include_half=True,
                                 include_bfloat16=True,
                                 include_bool=True,
                                 include_complex=True):
    return get_all_dtypes(
        include_half, include_bfloat16, include_bool, include_complex
        ) + list(_universal_types)

def get_all_fp_dtypes_and_universal(include_half=True, include_bfloat16=True):
    return get_all_fp_dtypes(include_half, include_bfloat16) + list(_universal_types)
