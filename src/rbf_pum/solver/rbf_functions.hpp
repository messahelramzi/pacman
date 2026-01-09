#pragma once

#include <Kokkos_Core.hpp>

#include "common/types.hpp"
#include "utils/utils.hpp"

namespace PACMAN {
namespace RbfPum {
class WendlandC0 {
public:
  KOKKOS_FORCEINLINE_FUNCTION
  fp_t constexpr Eval(const fp_t &r) const {
    const fp_t p = r * mRadiusInv;
    char mask =
        static_cast<char>(r >= fp_consts::zero() && p < fp_consts::one() &&
                          p >= fp_consts::zero());
    return static_cast<fp_t>(mask) *
           ((fp_consts::one() - p) * (fp_consts::one() - p));
  }
  inline fp_t constexpr EvalHost(const fp_t &r) const {
    const fp_t p = r * mRadiusInv;
    if (p >= fp_consts::one() || p < fp_consts::zero()) {
      return fp_consts::zero();
    }
    return std::pow(fp_consts::one() - p, 2);
  }

  inline void SetRadiusInv(const fp_t &radiusInv) {
    this->mRadiusInv = radiusInv;
  }

private:
  fp_t mRadiusInv;
};

class WendlandC2 {
public:
  KOKKOS_FORCEINLINE_FUNCTION
  fp_t Eval(const fp_t &r) const {
    const fp_t p = r * mRadiusInv;
    char mask =
        static_cast<char>(r >= fp_consts::zero() && p < fp_consts::one() &&
                          p >= fp_consts::zero());
    return static_cast<fp_t>(mask) *
           (Kokkos::pow(fp_consts::one() - p, 4) *
            Kokkos::fma(static_cast<fp_t>(4.0), p, fp_consts::one()));
  }

  inline fp_t constexpr EvalHost(const fp_t &r) const {
    const fp_t p = r * mRadiusInv;
    if (p >= 1 || p < 0) {
      return fp_consts::zero();
    }
    return std::pow(fp_consts::one() - p, 4) *
           Kokkos::fma(4.0, p, fp_consts::one());
  }

  inline void SetRadiusInv(const fp_t &radiusInv) {
    this->mRadiusInv = radiusInv;
  }

private:
  fp_t mRadiusInv;
};

class WendlandC4 {
public:
  KOKKOS_FORCEINLINE_FUNCTION
  fp_t Eval(const fp_t &r) const {
    const fp_t p = r * mRadiusInv;
    char mask =
        static_cast<char>(r >= fp_consts::zero() && p < fp_consts::one() &&
                          p >= fp_consts::zero());
    return static_cast<fp_t>(mask) *
           (Kokkos::pow(fp_consts::one() - p, 6) *
            (static_cast<fp_t>(35.0) * Kokkos::pow(p, 2) +
             Kokkos::fma(static_cast<fp_t>(18.0), p, static_cast<fp_t>(3.0))));
  }

  inline fp_t constexpr EvalHost(const fp_t &r) const {
    const fp_t p = r * mRadiusInv;
    if (p >= 1 || p < 0) {
      return fp_consts::zero();
    }
    return std::pow(fp_consts::one() - p, 6) *
           (35.0 * std::pow(p, 2) + Kokkos::fma(18.0, p, 3.0));
  }

  inline void SetRadiusInv(const fp_t &radiusInv) {
    this->mRadiusInv = radiusInv;
  };

private:
  fp_t mRadiusInv;
};

class WendlandC6 {
public:
  KOKKOS_FORCEINLINE_FUNCTION
  fp_t Eval(const fp_t &r) const {
    const fp_t p = r * mRadiusInv;
    char mask =
        static_cast<char>(r >= fp_consts::zero() && p < fp_consts::one() &&
                          p >= fp_consts::zero());
    return static_cast<fp_t>(mask) *
           (Kokkos::pow(fp_consts::one() - p, 8) *
            (static_cast<fp_t>(32.0) * Kokkos::pow(p, 3) +
             static_cast<fp_t>(25.0) * Kokkos::pow(p, 2) +
             Kokkos::fma(static_cast<fp_t>(8.0), p, fp_consts::one())));
  }

  inline fp_t constexpr EvalHost(const fp_t &r) const {
    const fp_t p = r * mRadiusInv;
    if (p >= fp_consts::one() || p < fp_consts::zero()) {
      return fp_consts::zero();
    }

    return std::pow(fp_consts::one() - p, 8) *
           (32.0 * std::pow(p, 3) + 25.0 * std::pow(p, 2) +
            Kokkos::fma(8.0, p, fp_consts::one()));
  }

  inline void SetRadiusInv(const fp_t &radiusInv) {
    this->mRadiusInv = radiusInv;
  }

private:
  fp_t mRadiusInv;
};

class WendlandC8 {
public:
  KOKKOS_FORCEINLINE_FUNCTION
  fp_t Eval(const fp_t &r) const {
    const fp_t p = r * mRadiusInv;
    char mask =
        static_cast<char>(r >= fp_consts::zero() && p < fp_consts::one() &&
                          p >= fp_consts::zero());
    return static_cast<fp_t>(mask) *
           (Kokkos::pow(fp_consts::one() - p, 10) *
                (static_cast<fp_t>(1287.0) * Kokkos::pow(p, 4) +
                 static_cast<fp_t>(1350.0) * Kokkos::pow(p, 3) +
                 static_cast<fp_t>(630.0) * Kokkos::pow(p, 2)) +
            Kokkos::fma(static_cast<fp_t>(150.0), p, static_cast<fp_t>(15.0)));
  }
  inline fp_t constexpr EvalHost(const fp_t &r) const {
    const fp_t p = r * mRadiusInv;
    if (p >= fp_consts::one() || p < fp_consts::zero()) {
      return fp_consts::zero();
    }
    return std::pow(fp_consts::one() - p, 10) *
           (1287.0 * std::pow(p, 4) + 1350.0 * std::pow(p, 3) +
            630.0 * std::pow(p, 2) + Kokkos::fma(150.0, p, 15.0));
  }

  inline void SetRadiusInv(const fp_t &radiusInv) {
    this->mRadiusInv = radiusInv;
  }

private:
  fp_t mRadiusInv;
};

} // namespace RbfPum
} // namespace PACMAN
