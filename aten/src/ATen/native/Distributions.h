#pragma once

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/Math.h>
#include <c10/macros/Macros.h>
#include <c10/util/MathConstants.h>
#include <c10/util/UniversalTypes-math.h>

// ROCM hcc doesn't work well with using std:: in kernel functions
#if defined(__CUDA_ARCH__)
#include <c10/cuda/CUDAMathCompat.h>
#define compat_exp c10::cuda::compat::exp
#define compat_ceil c10::cuda::compat::ceil
#define compat_floor c10::cuda::compat::floor
#define compat_log c10::cuda::compat::log
#define compat_pow c10::cuda::compat::pow
#define compat_sqrt c10::cuda::compat::sqrt
#define compat_tan c10::cuda::compat::tan
#define compat_abs c10::cuda::compat::abs
#define compat_log1p c10::cuda::compat::log1p
#elif defined(__HIPCC__)
#include <c10/hip/HIPMathCompat.h>
#define compat_exp c10::hip::compat::exp
#define compat_ceil c10::hip::compat::ceil
#define compat_floor c10::hip::compat::floor
#define compat_log c10::hip::compat::log
#define compat_pow c10::hip::compat::pow
#define compat_sqrt c10::hip::compat::sqrt
#define compat_tan c10::hip::compat::tan
#define compat_abs c10::hip::compat::abs
#define compat_log1p c10::hip::compat::log1p
#else
#define compat_exp std::exp
#define compat_ceil std::ceil
#define compat_floor std::floor
#define compat_log std::log
#define compat_pow std::pow
#define compat_sqrt std::sqrt
#define compat_tan std::tan
#define compat_abs std::abs
#define compat_log1p std::log1p
#endif

namespace {

#if !defined(__CUDA_ARCH__) && !defined(__HIPCC__)
// we cannot use std::isnan directly due to some incompatibility of
// gcc constexpr'ing and nvcc
using std::isnan;
#endif

// Here sampler_t should be function type scalar_t(void). For gpu
// "sampler" is a device function, but since ROCM doesn't have
// equivalent to nvstd::function, we use a template type parameter to
// capture it.
template<typename scalar_t, typename sampler_t>
struct BaseSampler {
  sampler_t sampler;
  C10_DEVICE BaseSampler(const sampler_t& sampler): sampler(sampler) {}
  C10_DEVICE scalar_t sample() {
    return sampler();
  }
};

// The function `sample_gamma` is
// is adapted from Numpy's distributions.c implementation.
// It is MIT licensed, so here is the copyright:

/* Copyright 2005 Robert Kern (robert.kern@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

template<typename scalar_t, typename accscalar_t, typename uniform_sampler_t, typename normal_sampler_t>
C10_DEVICE scalar_t sample_gamma(scalar_t alpha, BaseSampler<accscalar_t, uniform_sampler_t>& standard_uniform, BaseSampler<accscalar_t, normal_sampler_t>& standard_normal) {
  accscalar_t scale = 1.0f;

  // Boost alpha for higher acceptance probability.
  if (alpha < 1.0f) {
    if (alpha == 0.f) return 0.f;
    scale *= compat_pow(1 - standard_uniform.sample(), 1.0f / alpha);
    alpha += 1.0f;
  }

  // This implements the acceptance-rejection method of Marsaglia and Tsang (2000)
  // doi:10.1145/358407.358414
  const accscalar_t d = alpha - 1.0f / 3.0f;
  const accscalar_t c = 1.0f / compat_sqrt(9.0f * d);
  for (;;) {
    accscalar_t x, y;
    do {
      x = standard_normal.sample();
      y = 1.0f + c * x;
    } while (y <= 0);
    const accscalar_t v = y * y * y;
    const accscalar_t u = 1 - standard_uniform.sample();
    const accscalar_t xx = x * x;
    if (u < 1.0f - 0.0331f * xx * xx)
      return static_cast<scalar_t>(scale * d * v);
    if (compat_log(u) < 0.5f * xx + d * (1.0f - v + compat_log(v)))
      return static_cast<scalar_t>(scale * d * v);
  }
}

/* the functions stirling_approx_tail, binomial_inversion, and btrs are adapted
 * from TensorFlow's random_binomial_op.cc implementation. That code is under
 * copyright: 2019 The TensorFlow Authors.
 *
 * It was released under the Apache License, Version 2.0 (the "License"), available at:
 *    http://www.apache.org/licenses/LICENSE-2.0
 */

template<typename scalar_t>
C10_DEVICE scalar_t stirling_approx_tail(scalar_t k) {
  const static scalar_t kTailValues[] = {
    0.0810614667953272,
    0.0413406959554092,
    0.0276779256849983,
    0.02079067210376509,
    0.0166446911898211,
    0.0138761288230707,
    0.0118967099458917,
    0.0104112652619720,
    0.00925546218271273,
    0.00833056343336287
  };
  if (k <= 9) {
    return kTailValues[static_cast<size_t>(k)];
  }
  scalar_t kp1sq = (k + 1) * (k + 1);
  return (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / kp1sq) / kp1sq) / (k + 1);
}

template <>
C10_DEVICE c10::CFloatWithSubnormals stirling_approx_tail<c10::CFloatWithSubnormals>(
    c10::CFloatWithSubnormals k) {
  // CUDA does not support dynamic initialization in static variables, and
  // c10::CFloatWithSubnormals::CFloatWithSubnormals(float) is not constexpr, so
  // we're using the constexpr constructor instead
  static constexpr c10::CFloatWithSubnormals kTailValues[] = {
    c10::CFloatWithSubnormals(0x3DA6038E, c10::CFloatWithSubnormals::from_bits_t()), // 0.0810614667953272
    c10::CFloatWithSubnormals(0x3D2954DD, c10::CFloatWithSubnormals::from_bits_t()), // 0.0413406959554092
    c10::CFloatWithSubnormals(0x3CE2BCD1, c10::CFloatWithSubnormals::from_bits_t()), // 0.0276779256849983
    c10::CFloatWithSubnormals(0x3CAA5133, c10::CFloatWithSubnormals::from_bits_t()), // 0.02079067210376509
    c10::CFloatWithSubnormals(0x3C885A73, c10::CFloatWithSubnormals::from_bits_t()), // 0.0166446911898211
    c10::CFloatWithSubnormals(0x3C6358B4, c10::CFloatWithSubnormals::from_bits_t()), // 0.0138761288230707
    c10::CFloatWithSubnormals(0x3C42EA6B, c10::CFloatWithSubnormals::from_bits_t()), // 0.0118967099458917
    c10::CFloatWithSubnormals(0x3C2A9403, c10::CFloatWithSubnormals::from_bits_t()), // 0.0104112652619720
    c10::CFloatWithSubnormals(0x3C17A439, c10::CFloatWithSubnormals::from_bits_t()), // 0.00925546218271273
    c10::CFloatWithSubnormals(0x3C087CEA, c10::CFloatWithSubnormals::from_bits_t())  // 0.00833056343336287
  };
  if (k <= 9) {
    return kTailValues[static_cast<size_t>(k)];
  }
  c10::CFloatWithSubnormals kp1sq = (k + 1) * (k + 1);
  return (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / kp1sq) / kp1sq) / (k + 1);
}


template<typename scalar_t, typename accscalar_t, typename uniform_sampler_t>
C10_DEVICE scalar_t binomial_inversion(scalar_t count, scalar_t prob, BaseSampler<accscalar_t, uniform_sampler_t>& standard_uniform) {
  accscalar_t U;
  accscalar_t geom_sum = 0;
  scalar_t num_geom = 0;

  accscalar_t logprob = compat_log1p(-prob);

  while (1) {
    U = standard_uniform.sample();
    accscalar_t geom = compat_ceil(compat_log(U) / logprob);
    geom_sum += geom;
    if (geom_sum > count) {
      break;
    }
    num_geom = num_geom + 1;
  }
  return num_geom;
}

template<typename scalar_t, typename accscalar_t, typename uniform_sampler_t>
C10_DEVICE scalar_t btrs(scalar_t count, scalar_t prob, BaseSampler<accscalar_t, uniform_sampler_t>& standard_uniform) {
  scalar_t k;
  accscalar_t U, V, us;

  // This is spq in the paper.
  const accscalar_t stddev = compat_sqrt(count * prob * (1 - prob));

  // Other coefficients for Transformed Rejection sampling.
  const accscalar_t b = 1.15 + 2.53 * stddev;
  const accscalar_t a = -0.0873 + 0.0248 * b + 0.01 * prob;
  const accscalar_t c = count * prob + 0.5;
  const accscalar_t v_r = 0.92 - 4.2 / b;
  const accscalar_t r = prob / (1 - prob);

  const accscalar_t alpha = (2.83 + 5.1 / b) * stddev;
  const accscalar_t m = compat_floor((count + 1) * prob);

  while (1) {
    U = standard_uniform.sample() - 0.5;
    V = standard_uniform.sample();

    us = 0.5 - compat_abs(U);
    k = static_cast<scalar_t>(compat_floor((2 * a / us + b) * U + c));

    // Reject non-sensical answers.
    if (k < 0 || k > count) {
      continue;
    }
    // Region for which the box is tight, and we can return our calculated value.
    // This should happen 0.86 * v_r times. In the limit as n * p is large,
    // the acceptance rate converges to ~79% (and in the lower regime it is ~24%).
    if (us >= 0.07 && V <= v_r) {
      return k;
    }

    // This deviates from Hormann's BTRS algorithm, as there is a log missing.
    // For all (u, v) pairs outside of the bounding box, this calculates the
    // transformed-reject ratio.
    V = compat_log(V * alpha / (a / (us * us) + b));
    accscalar_t upperbound =
        ((m + 0.5) * compat_log((m + 1) / (r * (count - m + 1))) +
         (count + 1) * compat_log((count - m + 1) / (count - k + 1)) +
         (k + 0.5) * compat_log(r * (count - k + 1) / (k + 1)) +
         stirling_approx_tail<accscalar_t>(m) + stirling_approx_tail<accscalar_t>(count - m) -
         stirling_approx_tail<accscalar_t>(k) - stirling_approx_tail<accscalar_t>(count - k));

    if (V <= upperbound) {
      return k;
    }
  }
}

template<typename scalar_t, typename accscalar_t, typename uniform_sampler_t>
C10_DEVICE scalar_t sample_binomial(scalar_t count, scalar_t prob, BaseSampler<accscalar_t, uniform_sampler_t>& standard_uniform) {
  if (count <= 0.0 || prob <= 0.0) {
    return 0;
  } else if (prob >= 1.0) {
    return count;
  } else if (prob <= 0.5) {
    if (count * prob >= 10.0) {
      // btrs
      return btrs<scalar_t, accscalar_t, uniform_sampler_t>(count, prob, standard_uniform);
    } else {
      // binomial inversion
      return binomial_inversion<scalar_t, accscalar_t, uniform_sampler_t>(count, prob, standard_uniform);
    }
  } else if (prob > 0.5) {
    scalar_t qprob = 1.0 - prob;
    if (count * qprob >= 10.0) {
      // btrs
      return count - btrs<scalar_t, accscalar_t, uniform_sampler_t>(count, qprob, standard_uniform);
    } else {
      // count - binomial inversion
      return count - binomial_inversion<scalar_t, accscalar_t, uniform_sampler_t>(count, qprob, standard_uniform);
    }
  } else {
    // prob is nan?
    return static_cast<scalar_t>(NAN);
  }
}

/*
 * This function is derived from the implementation of the digamma function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library] in ATen/native/Math.h.
 */
template<typename scalar_t, typename accscalar_t>
C10_DEVICE static inline scalar_t digamma_one(scalar_t x) {
  constexpr accscalar_t PSI_10 = 2.25175258906672110764;
  if (x == 0) {
    return INFINITY;
  }
  accscalar_t additional_summand = 0;
  int x_is_integer = x == compat_floor(x);
  if (x < 0) {
    if (x_is_integer) {
      return INFINITY;
    }
    // it is more standard to write this as recursion, but
    // nvcc does not like that
    additional_summand = -c10::pi<scalar_t> /
        compat_tan(c10::pi<scalar_t> * x);
    x = 1 - x;
  }

  // Push x to be >= 10
  accscalar_t result = 0;
  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  if (x == 10) {
    return result + PSI_10 + additional_summand;
  }

  // Compute asymptotic digamma
  static const accscalar_t A[] = {
     8.33333333333333333333E-2,
    -2.10927960927960927961E-2,
     7.57575757575757575758E-3,
    -4.16666666666666666667E-3,
     3.96825396825396825397E-3,
    -8.33333333333333333333E-3,
     8.33333333333333333333E-2,
  };

  accscalar_t y = 0;
  if (x < 1.0e17f) {
    accscalar_t z = 1.0 / (x * x);
    y = z * polevl<accscalar_t>(z, A, 6);
  }
  return static_cast<scalar_t>(
      result + compat_log(x) - (0.5f / x) - y + additional_summand);
}

template <>
C10_HOST_DEVICE inline c10::CFloatWithSubnormals digamma_one<c10::CFloatWithSubnormals, c10::CFloatWithSubnormals>(c10::CFloatWithSubnormals x) {
  using scalar_t = c10::CFloatWithSubnormals;
  using accscalar_t = c10::CFloatWithSubnormals;
  const accscalar_t PSI_10 = 2.25175258906672110764;
  if (x == 0) {
    return INFINITY;
  }
  accscalar_t additional_summand = 0;
  int x_is_integer = x == compat_floor(x);
  if (x < 0) {
    if (x_is_integer) {
      return INFINITY;
    }
    // it is more standard to write this as recursion, but
    // nvcc does not like that
    additional_summand = -c10::detail::pi<scalar_t>() /
        compat_tan(c10::detail::pi<scalar_t>() * x);
    x = 1 - x;
  }

  // Push x to be >= 10
  accscalar_t result = 0;
  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  if (x == 10) {
    return result + PSI_10 + additional_summand;
  }

  // Compute asymptotic digamma
  // CUDA does not support dynamic initialization in static variables, and
  // c10::CFloatWithSubnormals::CFloatWithSubnormals(float) is not constexpr, so
  // we're using the constexpr constructor instead
  static constexpr accscalar_t A[] = {
    c10::CFloatWithSubnormals(0x3DAAAAAB, c10::CFloatWithSubnormals::from_bits_t()), //  8.33333333333333333333E-2
    c10::CFloatWithSubnormals(0xBCACCACD, c10::CFloatWithSubnormals::from_bits_t()), // -2.10927960927960927961E-2
    c10::CFloatWithSubnormals(0x3BF83E10, c10::CFloatWithSubnormals::from_bits_t()), //  7.57575757575757575758E-3
    c10::CFloatWithSubnormals(0xBB888889, c10::CFloatWithSubnormals::from_bits_t()), // -4.16666666666666666667E-3
    c10::CFloatWithSubnormals(0x3B820821, c10::CFloatWithSubnormals::from_bits_t()), //  3.96825396825396825397E-3
    c10::CFloatWithSubnormals(0xBC088889, c10::CFloatWithSubnormals::from_bits_t()), // -8.33333333333333333333E-3
    c10::CFloatWithSubnormals(0x3DAAAAAB, c10::CFloatWithSubnormals::from_bits_t())  //  8.33333333333333333333E-2
  };

  accscalar_t y = 0;
  if (x < 1.0e17f) {
    accscalar_t z = 1.0 / (x * x);
    y = z * polevl<accscalar_t>(z, A, 6);
  }
  return static_cast<scalar_t>(
      result + compat_log(x) - (0.5f / x) - y + additional_summand);
}

// Computes the reparameterized gradient -(d/dalpha cdf(x;alpha)) / pdf(x;alpha)
// for random number x drawn from a standard Gamma distribution Gamma(alpha).
template <typename scalar_t, typename accscalar_t>
C10_HOST_DEVICE scalar_t standard_gamma_grad_one(scalar_t alpha_, scalar_t x_) {
  // Use a Taylor series expansion for small x.
  accscalar_t x = static_cast<accscalar_t>(x_);
  accscalar_t alpha = static_cast<accscalar_t>(alpha_);
  if (x < 0.8f) {
    accscalar_t numer = 1;
    accscalar_t denom = alpha;
    auto series1 = numer / denom;
    auto series2 = numer / (denom * denom);
    for (int i = 1; i <= 5; ++i) {
      numer *= -x / static_cast<accscalar_t>(i);
      denom += 1;
      series1 += numer / denom;
      series2 += numer / (denom * denom);
    }
    const auto pow_x_alpha = compat_pow(x, alpha);
    const auto gamma_pdf = compat_pow(x, alpha - 1) * compat_exp(-x);
    const auto gamma_cdf = pow_x_alpha * series1;
    const auto gamma_cdf_alpha =
        (compat_log(x) - digamma_one<accscalar_t, accscalar_t>(alpha)) *
            gamma_cdf -
        pow_x_alpha * series2;
    const auto result = -gamma_cdf_alpha / gamma_pdf;
    return isnan(result) ? static_cast<scalar_t>( 0.f ) : static_cast<scalar_t>(result);
  }

  // Use a Rice saddle point expansion for large alpha.
  if (alpha > 8.0f) {
    if (0.9f * alpha <= x && x <= 1.1f * alpha) {
      const auto numer_1 = 1 + 24 * alpha * (1 + 12 * alpha);
      const auto numer_2 = 1440 * (alpha * alpha) + 6 * x * (53 - 120 * x)
          - 65 * x * x / alpha + alpha * (107 + 3600 * x);
      const auto denom = 1244160 * (alpha * alpha) * (alpha * alpha);
      return static_cast<scalar_t>(numer_1 * numer_2 / denom);
    }
    const auto denom = compat_sqrt(8 * alpha);
    const auto term2 = denom / (alpha - x);
    const auto term3 = compat_pow(
        x - alpha - alpha * compat_log(x / alpha),
        static_cast<accscalar_t>(-1.5));
    const auto term23 = (x < alpha) ? term2 - term3 : term2 + term3;
    const auto term1 = compat_log(x / alpha) * term23 -
        compat_sqrt(2 / alpha) * (alpha + x) / ((alpha - x) * (alpha - x));
    const auto stirling = 1 + 1 / (12 * alpha) * (1 + 1 / (24 * alpha));
    const auto numer = x * term1;
    return static_cast<scalar_t>(-stirling * numer / denom);
  }

  // Use a bivariate rational approximation to the reparameterized gradient.
  const auto u = compat_log(x / alpha);
  const auto v = compat_log(alpha);
  static const accscalar_t coef_uv[3][8] = {
    {0.16009398, -0.094634809, 0.025146376, -0.0030648343,
     1, 0.32668115, 0.10406089, 0.0014179084},
    {0.53487893, 0.1298071, 0.065735949, -0.0015649758,
     0.16639465, 0.020070113, -0.0035938915, -0.00058392623},
    {0.040121004, -0.0065914022, -0.0026286047, -0.0013441777,
     0.017050642, -0.0021309326, 0.00085092367, -1.5247877e-07},
  };
  accscalar_t coef_v[8];
  for (int i = 0; i < 8; ++ i) {
    coef_v[i] = coef_uv[0][i] + u * (coef_uv[1][i] + u * coef_uv[2][i]);
  }
  const auto p = coef_v[0] + v * (coef_v[1] + v * (coef_v[2] + v * coef_v[3]));
  const auto q = coef_v[4] + v * (coef_v[5] + v * (coef_v[6] + v * coef_v[7]));
  return static_cast<scalar_t>(compat_exp(p / q));
}

template <>
C10_HOST_DEVICE c10::CFloatWithSubnormals standard_gamma_grad_one<c10::CFloatWithSubnormals, c10::CFloatWithSubnormals>(
    c10::CFloatWithSubnormals alpha_, c10::CFloatWithSubnormals x_) {
  using scalar_t = c10::CFloatWithSubnormals;
  using accscalar_t = c10::CFloatWithSubnormals;
  // Use a Taylor series expansion for small x.
  accscalar_t x = static_cast<accscalar_t>(x_);
  accscalar_t alpha = static_cast<accscalar_t>(alpha_);
  if (x < 0.8f) {
    accscalar_t numer = 1;
    accscalar_t denom = alpha;
    auto series1 = numer / denom;
    auto series2 = numer / (denom * denom);
    for (int i = 1; i <= 5; ++i) {
      numer *= -x / static_cast<accscalar_t>(i);
      denom += 1;
      series1 += numer / denom;
      series2 += numer / (denom * denom);
    }
    const auto pow_x_alpha = compat_pow(x, alpha);
    const auto gamma_pdf = compat_pow(x, alpha - 1) * compat_exp(-x);
    const auto gamma_cdf = pow_x_alpha * series1;
    const auto gamma_cdf_alpha =
        (compat_log(x) - digamma_one<accscalar_t, accscalar_t>(alpha)) *
            gamma_cdf -
        pow_x_alpha * series2;
    const auto result = -gamma_cdf_alpha / gamma_pdf;
    return isnan(result) ? static_cast<scalar_t>( 0.f ) : static_cast<scalar_t>(result);
  }

  // Use a Rice saddle point expansion for large alpha.
  if (alpha > 8.0f) {
    if (0.9f * alpha <= x && x <= 1.1f * alpha) {
      const auto numer_1 = 1 + 24 * alpha * (1 + 12 * alpha);
      const auto numer_2 = 1440 * (alpha * alpha) + 6 * x * (53 - 120 * x)
          - 65 * x * x / alpha + alpha * (107 + 3600 * x);
      const auto denom = 1244160 * (alpha * alpha) * (alpha * alpha);
      return static_cast<scalar_t>(numer_1 * numer_2 / denom);
    }
    const auto denom = compat_sqrt(8 * alpha);
    const auto term2 = denom / (alpha - x);
    const auto term3 = compat_pow(
        x - alpha - alpha * compat_log(x / alpha),
        static_cast<accscalar_t>(-1.5));
    const auto term23 = (x < alpha) ? term2 - term3 : term2 + term3;
    const auto term1 = compat_log(x / alpha) * term23 -
        compat_sqrt(2 / alpha) * (alpha + x) / ((alpha - x) * (alpha - x));
    const auto stirling = 1 + 1 / (12 * alpha) * (1 + 1 / (24 * alpha));
    const auto numer = x * term1;
    return static_cast<scalar_t>(-stirling * numer / denom);
  }

  // Use a bivariate rational approximation to the reparameterized gradient.
  const auto u = compat_log(x / alpha);
  const auto v = compat_log(alpha);
  // CUDA does not support dynamic initialization in static variables, and
  // c10::CFloatWithSubnormals::CFloatWithSubnormals(float) is not constexpr, so
  // we're using the constexpr constructor instead
  static constexpr accscalar_t coef_uv[3][8] = {
    {
      c10::CFloatWithSubnormals(0x3E23EFAD, c10::CFloatWithSubnormals::from_bits_t()), //  0.16009398
      c10::CFloatWithSubnormals(0xBDC1CFE5, c10::CFloatWithSubnormals::from_bits_t()), // -0.094634809
      c10::CFloatWithSubnormals(0x3CCDFFC6, c10::CFloatWithSubnormals::from_bits_t()), //  0.025146376
      c10::CFloatWithSubnormals(0xBB48DB63, c10::CFloatWithSubnormals::from_bits_t()), // -0.0030648343
      c10::CFloatWithSubnormals(0x3F800000, c10::CFloatWithSubnormals::from_bits_t()), //  1
      c10::CFloatWithSubnormals(0x3EA742C0, c10::CFloatWithSubnormals::from_bits_t()), //  0.32668115
      c10::CFloatWithSubnormals(0x3DD51DE0, c10::CFloatWithSubnormals::from_bits_t()), //  0.10406089
      c10::CFloatWithSubnormals(0x3AB9D91C, c10::CFloatWithSubnormals::from_bits_t())  //  0.0014179084
    },
    {
      c10::CFloatWithSubnormals(0x3F08EDD3, c10::CFloatWithSubnormals::from_bits_t()), //  0.53487893
      c10::CFloatWithSubnormals(0x3E04EC27, c10::CFloatWithSubnormals::from_bits_t()), //  0.1298071
      c10::CFloatWithSubnormals(0x3D86A092, c10::CFloatWithSubnormals::from_bits_t()), //  0.065735949
      c10::CFloatWithSubnormals(0xBACD1FE0, c10::CFloatWithSubnormals::from_bits_t()), // -0.0015649758
      c10::CFloatWithSubnormals(0x3E2A635C, c10::CFloatWithSubnormals::from_bits_t()), //  0.16639465
      c10::CFloatWithSubnormals(0x3CA46A14, c10::CFloatWithSubnormals::from_bits_t()), //  0.020070113
      c10::CFloatWithSubnormals(0xBB6B877E, c10::CFloatWithSubnormals::from_bits_t()), // -0.0035938915
      c10::CFloatWithSubnormals(0xBA1912A0, c10::CFloatWithSubnormals::from_bits_t())  // -0.00058392623
    },
    {
      c10::CFloatWithSubnormals(0x3D2455EC, c10::CFloatWithSubnormals::from_bits_t()), //  0.040121004
      c10::CFloatWithSubnormals(0xBBD7FCB0, c10::CFloatWithSubnormals::from_bits_t()), // -0.0065914022
      c10::CFloatWithSubnormals(0xBB2C44AB, c10::CFloatWithSubnormals::from_bits_t()), // -0.0026286047
      c10::CFloatWithSubnormals(0xBAB02F1F, c10::CFloatWithSubnormals::from_bits_t()), // -0.0013441777
      c10::CFloatWithSubnormals(0x3C8BADCA, c10::CFloatWithSubnormals::from_bits_t()), //  0.017050642
      c10::CFloatWithSubnormals(0xBB0BA71E, c10::CFloatWithSubnormals::from_bits_t()), // -0.0021309326
      c10::CFloatWithSubnormals(0x3A5F1085, c10::CFloatWithSubnormals::from_bits_t()), //  0.00085092367
      c10::CFloatWithSubnormals(0xB423B90C, c10::CFloatWithSubnormals::from_bits_t())  // -1.5247877e-07
    }
  };
  accscalar_t coef_v[8];
  for (int i = 0; i < 8; ++ i) {
    coef_v[i] = coef_uv[0][i] + u * (coef_uv[1][i] + u * coef_uv[2][i]);
  }
  const auto p = coef_v[0] + v * (coef_v[1] + v * (coef_v[2] + v * coef_v[3]));
  const auto q = coef_v[4] + v * (coef_v[5] + v * (coef_v[6] + v * coef_v[7]));
  return static_cast<scalar_t>(compat_exp(p / q));
}

// Approximate reparameterized gradient of Beta(x,alpha,beta) wrt alpha.
// Assumes x is close to zero and uses a Taylor expansion.
template <typename scalar_t, typename accscalar_t>
C10_HOST_DEVICE static inline scalar_t _beta_grad_alpha_small(scalar_t x, scalar_t alpha, scalar_t beta) {
  const scalar_t factor = digamma_one<scalar_t, accscalar_t>(alpha)
                        - digamma_one<scalar_t, accscalar_t>(alpha + beta) - compat_log(x);
  scalar_t numer = 1;
  scalar_t series = numer / alpha * (factor + 1 / alpha);
  for (int i = 1; i <= 10; ++i) {
    scalar_t casted_i = static_cast<scalar_t>(i);
    numer *= (casted_i - beta) * x / casted_i;
    const scalar_t denom = alpha + casted_i;
    series += numer / denom * (factor + 1 / denom);
  }
  const scalar_t result = x * compat_pow(1 - x, -beta) * series;
  return isnan(result) ? static_cast<scalar_t>( 0.f ) : result;
}

// Approximate reparameterized gradient of Beta(x,alpha,beta) wrt beta.
// Assumes x is close to zero and uses a Taylor expansion.
template <typename scalar_t, typename accscalar_t>
C10_HOST_DEVICE static inline scalar_t _beta_grad_beta_small(scalar_t x, scalar_t alpha, scalar_t beta) {
  const scalar_t factor = digamma_one<scalar_t, accscalar_t>(alpha + beta) - digamma_one<scalar_t, accscalar_t>(beta);
  scalar_t numer = 1, betas = 1, dbetas = 0, series = factor / alpha;
  for (int i = 1; i <= 8; ++i) {
    scalar_t casted_i = static_cast<scalar_t>(i);
    numer *= -x / casted_i;
    dbetas = dbetas * (beta - casted_i) + betas;
    betas = betas * (beta - casted_i);
    series += numer / (alpha + casted_i) * (dbetas + factor * betas);
  }
  const scalar_t result = -compat_pow(1 - x, 1 - beta) * series;
  return isnan(result) ? static_cast<scalar_t>( 0.f ) : result;
}

// Approximate reparameterized gradient of Beta(x,alpha,beta) wrt alpha.
// Assumes alpha and beta are both large and uses a Rice saddle point expansion.
// To ensure numerical stability, this computation is performed at higher precision.
template<typename scalar_t, typename accscalar_t>
C10_DEVICE static inline scalar_t _beta_grad_alpha_mid(accscalar_t x, accscalar_t alpha, accscalar_t beta) {
  const accscalar_t total = alpha + beta;
  const accscalar_t mean = alpha / total;
  const accscalar_t std = compat_sqrt(alpha * beta / (total + 1)) / total;
  if (mean - 0.1 * std <= x && x <= mean + 0.1 * std) {
    // Avoid the singularity at x = mean.
    const accscalar_t poly = 47 * x * (beta * beta) * (beta * beta) + alpha * (
                           (43 + 20 * (16 + 27 * beta) * x) * (beta * beta) * beta + alpha * (
                           3 * (59 + 180 * beta - 90 * x) * (beta * beta) + alpha * (
                           (453 + 1620 * beta * (1 - x) - 455 * x) * beta + alpha * (
                           8 * (1 - x) * (135 * beta - 11)))));
    const accscalar_t prefactor_num = (1 + 12 * alpha) * (1 + 12 * beta) / (total * total);
    const accscalar_t prefactor_den = 12960 * alpha * alpha * alpha * beta * beta * (1 + 12 * total);
    return prefactor_num / (1 - x) * poly / prefactor_den;
  }
  const accscalar_t prefactor = -x / compat_sqrt(2 * alpha * beta / total);
  const accscalar_t stirling = (1 + 1 / (12 * alpha) + 1 / (288 * alpha * alpha))
                             * (1 + 1 / (12 * beta) + 1 / (288 * beta * beta))
                             / (1 + 1 / (12 * total) + 1 / (288 * total * total));
  const accscalar_t term1_num = 2 * (alpha * alpha) * (x - 1) + alpha * beta * (x - 1) - x * (beta * beta);
  const accscalar_t axbx = alpha * (x - 1) + beta * x;
  const accscalar_t term1_den = compat_sqrt(2 * alpha / beta) * compat_pow(total, static_cast<accscalar_t>(1.5f)) * axbx * axbx;
  const accscalar_t term1 = term1_num / term1_den;
  const accscalar_t term2 = 0.5f * compat_log(alpha / (total * x));
  const accscalar_t term3_num = compat_sqrt(8 * alpha * beta / total);
  const accscalar_t term3_den = beta * x + alpha * (x - 1);
  const accscalar_t term3 = term3_num / term3_den;
  const accscalar_t term4_base = beta * compat_log(beta / (total * (1 - x))) +
                               alpha * compat_log(alpha / (total * x));
  const accscalar_t term4 = compat_pow(term4_base, static_cast<accscalar_t>(-1.5f));
  const accscalar_t term1234 = term1 + term2 * (term3 + (x < mean ? term4 : -term4));
  return static_cast<scalar_t>(stirling * prefactor * term1234);
}

// Computes a scaled reparameterized gradient
//   -(d/dalpha cdf(x;alpha,beta)) / pdf(x;alpha,beta) / (1-x)
// for random number x drawn from a Beta distribution Beta(alpha,beta).
// This function inputs total=alpha+beta to make it easy to implement
// Dirichlet reparameterized gradients in terms of Betas.
template<typename scalar_t, typename accscalar_t>
C10_HOST_DEVICE static inline scalar_t dirichlet_grad_one(scalar_t x, scalar_t alpha, scalar_t total) {
  accscalar_t x_ = static_cast<accscalar_t>(x);
  accscalar_t alpha_ = static_cast<accscalar_t>(alpha);
  accscalar_t total_ = static_cast<accscalar_t>(total);

  const scalar_t beta = total - alpha;
  const accscalar_t beta_ = total_ - alpha_;
  const scalar_t boundary = total * x * (1 - x);

  // Use an asymptotic approximation for x close to 0.
  if (x <= 0.5f && boundary < 2.5f) {
    return _beta_grad_alpha_small<scalar_t, accscalar_t>(x, alpha, beta);
  }

  // Use an asymptotic approximation for x close to 1.
  if (x >= 0.5f && boundary < 0.75f) {
    return -_beta_grad_beta_small<scalar_t, accscalar_t>(1 - x, beta, alpha);
  }

  // Use an asymptotic approximation when alpha and (total - alpha) are both large.
  if (alpha > 6 && beta > 6) {
    return _beta_grad_alpha_mid<scalar_t, accscalar_t>(x_, alpha_, beta_);
  }

  // Use a rational correction to an analytic approximation.
  static const accscalar_t c[2][3][3][4] = {
    {{{1.003668233, -0.01061107488, -0.0657888334, 0.01201642863},
      {0.6336835991, -0.3557432599, 0.05486251648, -0.001465281033},
      {-0.03276231906, 0.004474107445, 0.002429354597, -0.0001557569013}},
     {{0.221950385, -0.3187676331, 0.01799915743, 0.01074823814},
      {-0.2951249643, 0.06219954479, 0.01535556598, 0.001550077057},
      {0.02155310298, 0.004170831599, 0.001292462449, 6.976601077e-05}},
     {{-0.05980841433, 0.008441916499, 0.01085618172, 0.002319392565},
      {0.02911413504, 0.01400243777, -0.002721828457, 0.000751041181},
      {0.005900514878, -0.001936558688, -9.495446725e-06, 5.385558597e-05}}},
    {{{1, -0.02924021934, -0.04438342661, 0.007285809825},
      {0.6357567472, -0.3473456711, 0.05454656494, -0.002407477521},
      {-0.03301322327, 0.004845219414, 0.00231480583, -0.0002307248149}},
     {{0.5925320577, -0.1757678135, 0.01505928619, 0.000564515273},
      {0.1014815858, -0.06589186703, 0.01272886114, -0.0007316646956},
      {-0.007258481865, 0.001096195486, 0.0003934994223, -4.12701925e-05}},
     {{0.06469649321, -0.0236701437, 0.002902096474, -5.896963079e-05},
      {0.001925008108, -0.002869809258, 0.0008000589141, -6.063713228e-05},
      {-0.0003477407336, 6.959756487e-05, 1.097287507e-05, -1.650964693e-06}}},
  };
  const accscalar_t u = compat_log(x_);
  const accscalar_t a = compat_log(alpha_) - u;
  const accscalar_t b = compat_log(total_) - a;
  const accscalar_t pow_u[3] = {1, u, u * u};
  const accscalar_t pow_a[3] = {1, a, a * a};
  accscalar_t p = 0.0;
  accscalar_t q = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const accscalar_t ua = pow_u[i] * pow_a[j];
      p += ua * (c[0][i][j][0] + b * (c[0][i][j][1] + b * (c[0][i][j][2] + b * c[0][i][j][3])));
      q += ua * (c[1][i][j][0] + b * (c[1][i][j][1] + b * (c[1][i][j][2] + b * c[1][i][j][3])));
    }
  }
  const accscalar_t approx = x_ * (digamma_one<scalar_t, accscalar_t>(total_) - digamma_one<scalar_t, accscalar_t>(alpha_)) / beta_;
  return static_cast<scalar_t>(p / q * approx);
}

template <>
C10_HOST_DEVICE inline c10::CFloatWithSubnormals dirichlet_grad_one<c10::CFloatWithSubnormals, c10::CFloatWithSubnormals>(
    c10::CFloatWithSubnormals x,
    c10::CFloatWithSubnormals alpha,
    c10::CFloatWithSubnormals total) {
  using scalar_t = c10::CFloatWithSubnormals;
  using accscalar_t = c10::CFloatWithSubnormals;
  accscalar_t x_ = static_cast<accscalar_t>(x);
  accscalar_t alpha_ = static_cast<accscalar_t>(alpha);
  accscalar_t total_ = static_cast<accscalar_t>(total);

  const scalar_t beta = total - alpha;
  const accscalar_t beta_ = total_ - alpha_;
  const scalar_t boundary = total * x * (1 - x);

  // Use an asymptotic approximation for x close to 0.
  if (x <= 0.5f && boundary < 2.5f) {
    return _beta_grad_alpha_small<scalar_t, accscalar_t>(x, alpha, beta);
  }

  // Use an asymptotic approximation for x close to 1.
  if (x >= 0.5f && boundary < 0.75f) {
    return -_beta_grad_beta_small<scalar_t, accscalar_t>(1 - x, beta, alpha);
  }

  // Use an asymptotic approximation when alpha and (total - alpha) are both large.
  if (alpha > 6 && beta > 6) {
    return _beta_grad_alpha_mid<scalar_t, accscalar_t>(x_, alpha_, beta_);
  }

  // Use a rational correction to an analytic approximation.
  // CUDA does not support dynamic initialization in static variables, and
  // c10::CFloatWithSubnormals::CFloatWithSubnormals(float) is not constexpr, so
  // we're using the constexpr constructor instead
  static constexpr accscalar_t c[2][3][3][4] = {
    {
      {
        {
          c10::CFloatWithSubnormals(0x3F807833, c10::CFloatWithSubnormals::from_bits_t()), // 1.003668233
          c10::CFloatWithSubnormals(0xBC2DDA13, c10::CFloatWithSubnormals::from_bits_t()), // -0.01061107488
          c10::CFloatWithSubnormals(0xBD86BC4C, c10::CFloatWithSubnormals::from_bits_t()), // -0.0657888334
          c10::CFloatWithSubnormals(0x3C44E08E, c10::CFloatWithSubnormals::from_bits_t()), // 0.01201642863
        },
        {
          c10::CFloatWithSubnormals(0x3F223917, c10::CFloatWithSubnormals::from_bits_t()), // 0.6336835991
          c10::CFloatWithSubnormals(0xBEB623FB, c10::CFloatWithSubnormals::from_bits_t()), // -0.3557432599
          c10::CFloatWithSubnormals(0x3D60B785, c10::CFloatWithSubnormals::from_bits_t()), // 0.05486251648
          c10::CFloatWithSubnormals(0xBAC00EAC, c10::CFloatWithSubnormals::from_bits_t()), // -0.001465281033
        },
        {
          c10::CFloatWithSubnormals(0xBD0631C8, c10::CFloatWithSubnormals::from_bits_t()), // -0.03276231906
          c10::CFloatWithSubnormals(0x3B929B89, c10::CFloatWithSubnormals::from_bits_t()), // 0.004474107445
          c10::CFloatWithSubnormals(0x3B1F35CF, c10::CFloatWithSubnormals::from_bits_t()), // 0.002429354597
          c10::CFloatWithSubnormals(0xB92352AD, c10::CFloatWithSubnormals::from_bits_t()), // -0.0001557569013
        },
      },
      {
        {
          c10::CFloatWithSubnormals(0x3E6346F6, c10::CFloatWithSubnormals::from_bits_t()), // 0.221950385
          c10::CFloatWithSubnormals(0xBEA33583, c10::CFloatWithSubnormals::from_bits_t()), // -0.3187676331
          c10::CFloatWithSubnormals(0x3C9372F8, c10::CFloatWithSubnormals::from_bits_t()), // 0.01799915743
          c10::CFloatWithSubnormals(0x3C301961, c10::CFloatWithSubnormals::from_bits_t()), // 0.01074823814
        },
        {
          c10::CFloatWithSubnormals(0xBE971A9F, c10::CFloatWithSubnormals::from_bits_t()), // -0.2951249643
          c10::CFloatWithSubnormals(0x3D7EC4F3, c10::CFloatWithSubnormals::from_bits_t()), // 0.06219954479
          c10::CFloatWithSubnormals(0x3C7B95E9, c10::CFloatWithSubnormals::from_bits_t()), // 0.01535556598
          c10::CFloatWithSubnormals(0x3ACB2BF5, c10::CFloatWithSubnormals::from_bits_t()), // 0.001550077057
        },
        {
          c10::CFloatWithSubnormals(0x3CB09022, c10::CFloatWithSubnormals::from_bits_t()), // 0.02155310298
          c10::CFloatWithSubnormals(0x3B88AB79, c10::CFloatWithSubnormals::from_bits_t()), // 0.004170831599
          c10::CFloatWithSubnormals(0x3AA967D8, c10::CFloatWithSubnormals::from_bits_t()), // 0.001292462449
          c10::CFloatWithSubnormals(0x38924F58, c10::CFloatWithSubnormals::from_bits_t()), // 6.976601077e-05
        },
      },
      {
        {
          c10::CFloatWithSubnormals(0xBD74F9AB, c10::CFloatWithSubnormals::from_bits_t()), // -0.05980841433
          c10::CFloatWithSubnormals(0x3C0A4FF7, c10::CFloatWithSubnormals::from_bits_t()), // 0.008441916499
          c10::CFloatWithSubnormals(0x3C31DE20, c10::CFloatWithSubnormals::from_bits_t()), // 0.01085618172
          c10::CFloatWithSubnormals(0x3B1800F3, c10::CFloatWithSubnormals::from_bits_t()), // 0.002319392565
        },
        {
          c10::CFloatWithSubnormals(0x3CEE80C4, c10::CFloatWithSubnormals::from_bits_t()), // 0.02911413504
          c10::CFloatWithSubnormals(0x3C656A7B, c10::CFloatWithSubnormals::from_bits_t()), // 0.01400243777
          c10::CFloatWithSubnormals(0xBB3260B4, c10::CFloatWithSubnormals::from_bits_t()), // -0.002721828457
          c10::CFloatWithSubnormals(0x3A44E185, c10::CFloatWithSubnormals::from_bits_t()), // 0.000751041181
        },
        {
          c10::CFloatWithSubnormals(0x3BC1591B, c10::CFloatWithSubnormals::from_bits_t()), // 0.005900514878
          c10::CFloatWithSubnormals(0xBAFDD420, c10::CFloatWithSubnormals::from_bits_t()), // -0.001936558688
          c10::CFloatWithSubnormals(0xB71F4EA2, c10::CFloatWithSubnormals::from_bits_t()), // -9.495446725e-06
          c10::CFloatWithSubnormals(0x3861E2FF, c10::CFloatWithSubnormals::from_bits_t()), // 5.385558597e-05
        },
      },
    },
    {
      {
        {
          c10::CFloatWithSubnormals(0x3F800000, c10::CFloatWithSubnormals::from_bits_t()), // 1
          c10::CFloatWithSubnormals(0xBCEF892F, c10::CFloatWithSubnormals::from_bits_t()), // -0.02924021934
          c10::CFloatWithSubnormals(0xBD35CB65, c10::CFloatWithSubnormals::from_bits_t()), // -0.04438342661
          c10::CFloatWithSubnormals(0x3BEEBDCD, c10::CFloatWithSubnormals::from_bits_t()), // 0.007285809825
        },
        {
          c10::CFloatWithSubnormals(0x3F22C0F4, c10::CFloatWithSubnormals::from_bits_t()), // 0.6357567472
          c10::CFloatWithSubnormals(0xBEB1D74B, c10::CFloatWithSubnormals::from_bits_t()), // -0.3473456711
          c10::CFloatWithSubnormals(0x3D5F6C38, c10::CFloatWithSubnormals::from_bits_t()), // 0.05454656494
          c10::CFloatWithSubnormals(0xBB1DC6C5, c10::CFloatWithSubnormals::from_bits_t()), // -0.002407477521
        },
        {
          c10::CFloatWithSubnormals(0xBD0738E0, c10::CFloatWithSubnormals::from_bits_t()), // -0.03301322327
          c10::CFloatWithSubnormals(0x3B9EC4A5, c10::CFloatWithSubnormals::from_bits_t()), // 0.004845219414
          c10::CFloatWithSubnormals(0x3B17B3FF, c10::CFloatWithSubnormals::from_bits_t()), // 0.00231480583
          c10::CFloatWithSubnormals(0xB971EEB9, c10::CFloatWithSubnormals::from_bits_t()), // -0.0002307248149
        },
      },
      {
        {
          c10::CFloatWithSubnormals(0x3F17B02E, c10::CFloatWithSubnormals::from_bits_t()), // 0.5925320577
          c10::CFloatWithSubnormals(0xBE33FC7A, c10::CFloatWithSubnormals::from_bits_t()), // -0.1757678135
          c10::CFloatWithSubnormals(0x3C76BB39, c10::CFloatWithSubnormals::from_bits_t()), // 0.01505928619
          c10::CFloatWithSubnormals(0x3A13FBFB, c10::CFloatWithSubnormals::from_bits_t()), // 0.000564515273
        },
        {
          c10::CFloatWithSubnormals(0x3DCFD594, c10::CFloatWithSubnormals::from_bits_t()), // 0.1014815858
          c10::CFloatWithSubnormals(0xBD86F251, c10::CFloatWithSubnormals::from_bits_t()), // -0.06589186703
          c10::CFloatWithSubnormals(0x3C508CB7, c10::CFloatWithSubnormals::from_bits_t()), // 0.01272886114
          c10::CFloatWithSubnormals(0xBA3FCD30, c10::CFloatWithSubnormals::from_bits_t()), // -0.0007316646956
        },
        {
          c10::CFloatWithSubnormals(0xBBEDD88F, c10::CFloatWithSubnormals::from_bits_t()), // -0.007258481865
          c10::CFloatWithSubnormals(0x3A8FAE38, c10::CFloatWithSubnormals::from_bits_t()), // 0.001096195486
          c10::CFloatWithSubnormals(0x39CE4E99, c10::CFloatWithSubnormals::from_bits_t()), // 0.0003934994223
          c10::CFloatWithSubnormals(0xB82D1988, c10::CFloatWithSubnormals::from_bits_t()), // -4.12701925e-05
        },
      },
      {
        {
          c10::CFloatWithSubnormals(0x3D847F98, c10::CFloatWithSubnormals::from_bits_t()), // 0.06469649321
          c10::CFloatWithSubnormals(0xBCC1E7E4, c10::CFloatWithSubnormals::from_bits_t()), // -0.0236701437
          c10::CFloatWithSubnormals(0x3B3E3119, c10::CFloatWithSubnormals::from_bits_t()), // 0.002902096474
          c10::CFloatWithSubnormals(0xB8775629, c10::CFloatWithSubnormals::from_bits_t()), // -5.896963079e-05
        },
        {
          c10::CFloatWithSubnormals(0x3AFC508E, c10::CFloatWithSubnormals::from_bits_t()), // 0.001925008108
          c10::CFloatWithSubnormals(0xBB3C1369, c10::CFloatWithSubnormals::from_bits_t()), // -0.002869809258
          c10::CFloatWithSubnormals(0x3A51BB0B, c10::CFloatWithSubnormals::from_bits_t()), // 0.0008000589141
          c10::CFloatWithSubnormals(0xB87E54A0, c10::CFloatWithSubnormals::from_bits_t()), // -6.063713228e-05
        },
        {
          c10::CFloatWithSubnormals(0xB9B650F9, c10::CFloatWithSubnormals::from_bits_t()), // -0.0003477407336
          c10::CFloatWithSubnormals(0x3891F4E8, c10::CFloatWithSubnormals::from_bits_t()), // 6.959756487e-05
          c10::CFloatWithSubnormals(0x37381824, c10::CFloatWithSubnormals::from_bits_t()), // 1.097287507e-05
          c10::CFloatWithSubnormals(0xB5DD96B7, c10::CFloatWithSubnormals::from_bits_t()), // -1.650964693e-06
        },
      },
    },
  };
  const accscalar_t u = compat_log(x_);
  const accscalar_t a = compat_log(alpha_) - u;
  const accscalar_t b = compat_log(total_) - a;
  const accscalar_t pow_u[3] = {1, u, u * u};
  const accscalar_t pow_a[3] = {1, a, a * a};
  accscalar_t p = 0.0;
  accscalar_t q = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const accscalar_t ua = pow_u[i] * pow_a[j];
      p += ua * (c[0][i][j][0] + b * (c[0][i][j][1] + b * (c[0][i][j][2] + b * c[0][i][j][3])));
      q += ua * (c[1][i][j][0] + b * (c[1][i][j][1] + b * (c[1][i][j][2] + b * c[1][i][j][3])));
    }
  }
  const accscalar_t approx = x_ * (digamma_one<scalar_t, accscalar_t>(total_) - digamma_one<scalar_t, accscalar_t>(alpha_)) / beta_;
  return static_cast<scalar_t>(p / q * approx);
}

} // namespace
