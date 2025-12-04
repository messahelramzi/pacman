#pragma once

#include <Kokkos_Core.hpp>

#include "utils/utils.hpp"

template <class RbfPumFPType>
class WendlandC0
{
public:
    KOKKOS_FORCEINLINE_FUNCTION
    RbfPumFPType constexpr eval(const RbfPumFPType& r) const
    {
        const RbfPumFPType p = r * _r_inv;
        char mask = static_cast<char>(r >= 0.0 && p < 1.0 && p >= 0.0);
        return static_cast<RbfPumFPType>(mask)
            * ((static_cast<RbfPumFPType>(1.0) - p)
               * (static_cast<RbfPumFPType>(1.0) - p));
    }
    __forceinline__
        RbfPumFPType constexpr eval_host(const RbfPumFPType& r) const
    {
        const RbfPumFPType p = r * _r_inv;
        if (p >= 1 || p < 0)
        {
            return 0.0;
        }
        return std::pow(1.0 - p, 2);
    }

    __forceinline__ void set_r_inv(const RbfPumFPType& r_inv)
    {
        this->_r_inv = r_inv;
    }

private:
    RbfPumFPType _r_inv;
};

template <class RbfPumFPType>
class WendlandC2
{
public:
    KOKKOS_FORCEINLINE_FUNCTION
    RbfPumFPType constexpr eval(const RbfPumFPType& r) const
    {
        const RbfPumFPType p = r * _r_inv;
        char mask = static_cast<char>(r >= 0.0 && p < 1.0 && p >= 0.0);
        return static_cast<RbfPumFPType>(mask)
            * (Kokkos::pow(static_cast<RbfPumFPType>(1.0) - p, 4)
               * rbfpum_fma(static_cast<RbfPumFPType>(4.0), p,
                            static_cast<RbfPumFPType>(1.0)));
    }

    __forceinline__
        RbfPumFPType constexpr eval_host(const RbfPumFPType& r) const
    {
        const RbfPumFPType p = r * _r_inv;
        if (p >= 1 || p < 0)
        {
            return 0.0;
        }
        return std::pow(1.0 - p, 4) * rbfpum_fma(4.0, p, 1.0);
    }

    __forceinline__ void set_r_inv(const RbfPumFPType& r_inv)
    {
        this->_r_inv = r_inv;
    }

private:
    RbfPumFPType _r_inv;
};

template <class RbfPumFPType>
class WendlandC4
{
public:
    KOKKOS_FORCEINLINE_FUNCTION
    RbfPumFPType constexpr eval(const RbfPumFPType& r) const
    {
        const RbfPumFPType p = r * _r_inv;
        char mask = static_cast<char>(r >= 0.0 && p < 1.0 && p >= 0.0);
        return static_cast<RbfPumFPType>(mask)
            * (Kokkos::pow(static_cast<RbfPumFPType>(1.0) - p, 6)
               * (static_cast<RbfPumFPType>(35.0) * Kokkos::pow(p, 2)
                  + rbfpum_fma(static_cast<RbfPumFPType>(18.0), p,
                               static_cast<RbfPumFPType>(3.0))));
    }

    __forceinline__
        RbfPumFPType constexpr eval_host(const RbfPumFPType& r) const
    {
        const RbfPumFPType p = r * _r_inv;
        if (p >= 1 || p < 0)
        {
            return 0.0;
        }
        return std::pow(1.0 - p, 6)
            * (35 * std::pow(p, 2) + rbfpum_fma(18.0, p, 3.0));
    }

    __forceinline__ void set_r_inv(const RbfPumFPType& r_inv)
    {
        this->_r_inv = r_inv;
    }

private:
    RbfPumFPType _r_inv;
};

template <class RbfPumFPType>
class WendlandC6
{
public:
    KOKKOS_FORCEINLINE_FUNCTION
    RbfPumFPType constexpr eval(const RbfPumFPType& r) const
    {
        const RbfPumFPType p = r * _r_inv;
        char mask = static_cast<char>(r >= 0.0 && p < 1.0 && p >= 0.0);
        return static_cast<RbfPumFPType>(mask)
            * (Kokkos::pow(static_cast<RbfPumFPType>(1.0) - p, 8)
               * (static_cast<RbfPumFPType>(32.0) * Kokkos::pow(p, 3)
                  + static_cast<RbfPumFPType>(25.0) * Kokkos::pow(p, 2)
                  + rbfpum_fma(static_cast<RbfPumFPType>(8.0), p,
                               static_cast<RbfPumFPType>(1.0))));
    }

    __forceinline__
        RbfPumFPType constexpr eval_host(const RbfPumFPType& r) const
    {
        const RbfPumFPType p = r * _r_inv;
        if (p >= 1 || p < 0)
        {
            return 0.0;
        }

        return std::pow(1.0 - p, 8)
            * (32.0 * std::pow(p, 3) + 25.0 * std::pow(p, 2)
               + rbfpum_fma(8.0, p, 1.0));
    }

    __forceinline__ void set_r_inv(const RbfPumFPType& r_inv)
    {
        this->_r_inv = r_inv;
    }

private:
    RbfPumFPType _r_inv;
};

template <class RbfPumFPType>
class WendlandC8
{
public:
    KOKKOS_FORCEINLINE_FUNCTION
    RbfPumFPType constexpr eval(const RbfPumFPType& r) const
    {
        const RbfPumFPType p = r * _r_inv;
        char mask = static_cast<char>(r >= 0.0 && p < 1.0 && p >= 0.0);
        return static_cast<RbfPumFPType>(mask)
            * (Kokkos::pow(static_cast<RbfPumFPType>(1.0) - p, 10)
                   * (static_cast<RbfPumFPType>(1287.0) * Kokkos::pow(p, 4)
                      + static_cast<RbfPumFPType>(1350.0) * Kokkos::pow(p, 3)
                      + static_cast<RbfPumFPType>(630.0) * Kokkos::pow(p, 2))
               + rbfpum_fma(static_cast<RbfPumFPType>(150.0), p,
                            static_cast<RbfPumFPType>(15.0)));
    }

    __forceinline__
        RbfPumFPType constexpr eval_host(const RbfPumFPType& r) const
    {
        const RbfPumFPType p = r * _r_inv;
        if (p >= 1 || p < 0)
        {
            return 0.0;
        }
        return std::pow(1.0 - p, 10)
            * (1287.0 * std::pow(p, 4) + 1350.0 * std::pow(p, 3)
               + 630.0 * std::pow(p, 2) + rbfpum_fma(150.0, p, 15.0));
    }

    __forceinline__ void set_r_inv(const RbfPumFPType& r_inv)
    {
        this->_r_inv = r_inv;
    }

private:
    RbfPumFPType _r_inv;
};
